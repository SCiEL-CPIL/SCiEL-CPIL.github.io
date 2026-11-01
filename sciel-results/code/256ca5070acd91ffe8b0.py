"""Minimal logarithmic energy of 150 points on the sphere.

Design: multiple well-dispersed restarts (Fibonacci / Halton / jittered),
each refined by a Riemannian conjugate-gradient pass (fast, robust) and then a
projected BFGS quasi-Newton local solve (fast, tight) for the final polish.
"""
import time
import math
import numpy as np

N = 150
_DEADLINE = 150.0
_MARGIN = 5.0


# ---------------------------------------------------------------------------
# Geometry / objective
# ---------------------------------------------------------------------------
def _normalize(p):
    """Project each row of (n,3) onto the unit sphere."""
    p = np.asarray(p, dtype=float)
    nrm = np.linalg.norm(p, axis=1, keepdims=True)
    nrm[nrm < 1e-15] = 1.0
    return p / nrm


def _energy_and_grad(p):
    """Logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) and its
    Riemannian (tangent-space) gradient on the unit sphere.

    E = -sum_{i<j} log d_ij.  Euclidean gradient w.r.t. x_i is
        g_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    which is projected to the tangent plane here.  The energy value is
    invariant under the tangent projection (constraint), so the same
    projected gradient drives both the CG and the quasi-Newton polish.
    """
    n = p.shape[0]
    diff = p[:, None, :] - p[None, :, :]              # (n,n,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)         # (n,n)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 1.0)
    w = np.zeros_like(d2)
    np.fill_diagonal(w, 0.0)
    w[d2 > 0.0] = 1.0 / d2[d2 > 0.0]

    # Euclidean gradient: g_i = -sum_j w_ij (x_i - x_j) = sum_j w_ij x_j - x_i sum_j w_ij
    grad = (w @ p) - p * w.sum(axis=1)[:, None]

    # Tangent projection
    inner = np.sum(grad * p, axis=1, keepdims=True)
    grad = grad - inner * p

    iu = np.triu_indices(n, k=1)
    dd = np.maximum(d[iu], 1e-15)
    energy = -np.sum(np.log(dd))
    return energy, grad


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------
def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _halton_sphere(n):
    def halton(index, base):
        f, r, idx = 1.0, 0.0, index
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        return r

    pts = np.zeros((n, 3))
    for i in range(n):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        z = 2.0 * v - 1.0
        theta = 2.0 * np.pi * u
        r = np.sqrt(max(0.0, 1.0 - z * z))
        pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return _normalize(pts)


def _jittered_fib(seed):
    rng = np.random.default_rng(seed)
    base = _fibonacci_sphere(N)
    return _normalize(base + rng.normal(0.0, 0.15, size=(N, 3)))


# ---------------------------------------------------------------------------
# Riemannian conjugate gradient (fast, robust)
# ---------------------------------------------------------------------------
def _riemannian_cg(x0, time_budget, max_iter=3000):
    x = _normalize(x0.copy())
    f, g = _energy_and_grad(x)
    best_x, best_e = x.copy(), f

    def exp_map(x, v):
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        nv[nv < 1e-15] = 1.0
        return x * np.cos(nv) + v * (np.sin(nv) / nv)

    g_r = g.copy()
    d = -g_r.copy()
    start = time.time()

    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        # backtracking line search along the geodesic
        step, f_new = 1.0, np.inf
        for _ls in range(30):
            x_new = _normalize(exp_map(x, step * d))
            f_new, _ = _energy_and_grad(x_new)
            if f_new <= f + 1e-4 * step * np.sum(g_r * d):
                break
            step *= 0.5
        g_new = _energy_and_grad(x_new)[1]
        g_new_r = g_new
        if f_new < best_e:
            best_e, best_x = f_new, x_new.copy()
        if np.linalg.norm(g_new_r) < 1e-9:
            break
        # Polak-Ribiere beta with restart on ascent
        beta = 0.0 if np.sum(g_new_r * d) > 0 else max(
            0.0, np.sum(g_new_r * (g_new_r - g_r)) / max(np.sum(g_r * g_r), 1e-15))
        d = -g_new_r + beta * d
        # re-project direction to tangent space
        d -= np.sum(d * x_new, axis=1, keepdims=True) * x_new
        x, f, g_r = x_new, f_new, g_new_r

    return _normalize(best_x), best_e


# ---------------------------------------------------------------------------
# Projected BFGS quasi-Newton polish (fast, tight local solve)
# ---------------------------------------------------------------------------
def _bfgs_polish(x0, time_budget, max_iter=400):
    x = _normalize(x0.copy())
    f, g = _energy_and_grad(x)
    best_x, best_e = x.copy(), f
    n = x.shape[0]
    H = np.eye(n)  # positive-definite Hessian approximation (tangent space)
    start = time.time()

    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        p = -H @ g
        p -= np.sum(p * x, axis=1, keepdims=True) * x   # tangent direction
        if np.linalg.norm(p) < 1e-14:
            break
        # backtracking line search on the geodesic
        step = 1.0
        x_new, f_new, g_new = x.copy(), f, g
        for _ls in range(40):
            cand = _normalize(x + step * p)
            fc, gc = _energy_and_grad(cand)
            if fc <= f + 1e-4 * step * np.sum(g * p):
                x_new, f_new, g_new = cand, fc, gc
                break
            step *= 0.5
        s = x_new - x
        y = g_new - g
        sy = np.sum(s * y)
        if sy > 1e-12:
            # BFGS update, kept positive definite
            H = (H - np.outer(H @ s, H @ s) / np.sum(s * H @ s)) \
                + np.outer(s, s) / sy
        if f_new < best_e:
            best_e, best_x = f_new, x_new.copy()
        if np.linalg.norm(g_new) < 1e-10:
            break
        x, f, g = x_new, f_new, g_new

    return _normalize(best_x), best_e


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    budget = _DEADLINE - _MARGIN - (time.time() - t0)
    if budget <= 0:
        return _fibonacci_sphere(N)

    best_x, best_e = None, np.inf

    def refine(x0, tag):
        nonlocal best_x, best_e
        elapsed = time.time() - t0
        rem = _DEADLINE - _MARGIN - elapsed
        if rem < 1.0:
            return
        # CG first (cheap, robust), then BFGS polish (fast, tight)
        xc, ec = _riemannian_cg(x0, rem * 0.6)
        if ec < best_e:
            best_e, best_x = ec, xc
        rem = _DEADLINE - _MARGIN - (time.time() - t0)
        if rem > 0.8:
            xb, eb = _bfgs_polish(xc, rem)
            if eb < best_e:
                best_e, best_x = eb, xb
        print(f"[{tag}] energy={best_e:.10f} t={time.time()-t0:.1f}s", flush=True)

    # Core restarts (always run if time permits)
    refine(_fibonacci_sphere(N), "fib")
    refine(_halton_sphere(N), "halton")

    # Extra restarts only if measured time permits (jittered escapes)
    for seed in (1, 2, 3):
        if _DEADLINE - _MARGIN - (time.time() - t0) < 4.0:
            break
        refine(_jittered_fib(seed), f"jit{seed}")

    if best_x is None:
        return _fibonacci_sphere(N)

    best_x = _normalize(best_x)
    if not np.all(np.isfinite(best_x)) or np.any(np.linalg.norm(best_x, axis=1) < 1e-12):
        return _fibonacci_sphere(N)
    return best_x
