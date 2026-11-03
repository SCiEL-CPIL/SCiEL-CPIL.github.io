"""Fifty equal spheres in a unit cube: maximize the common radius.

Fresh design: vectorized gradient ascent on a soft-min (log-sum-exp) surrogate of
the true radius, annealed over beta, with jittered restarts and a final polish.
The returned configuration is the best one measured by its true min-distance
(positions determine the score, not the soft value), so the result is robust to
any late-stage numerical instability.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0
MARGIN = 8.0          # seconds of safety before the 150 s kill
RNG_SEED = 20240517


def _true_radius(c):
    """Exact radius for a (N,3) array: min(pair half-dist, wall distances)."""
    c = np.asarray(c)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    iu = np.triu_indices(N, 1)
    pd = dist[iu]
    if pd.size:
        m = float(pd.min())
    else:
        m = np.inf
    wall = 0.5 - np.abs(c)
    return min(m * 0.5, float(wall.min()))


def _project(c):
    """Clamp centres into the closed cube [-0.5, 0.5]^3 (safety, never worsens)."""
    return np.clip(np.asarray(c, dtype=float), -0.5, 0.5)


def _soft_and_grad(c, beta):
    """Soft-min radius surrogate and its gradient. Returns (r_soft, grad, dist)."""
    c = np.asarray(c, dtype=float)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, 1)
    pd = dist[iu]
    wall = 0.5 - np.abs(c)

    vals = np.concatenate([pd, wall.ravel()])
    logsumexp = np.logaddexp.reduce(np.log(vals)) - np.log(vals.size)
    r_soft = float(np.exp(logsumexp))

    p_w = np.exp(np.log(pd) - logsumexp) / vals.size   # pair weights
    w_w = np.exp(np.log(wall.ravel()) - logsumexp) / vals.size  # wall weights

    # pair contribution: grad_i = (c_i - c_j) * p_w_ij / d_ij  (unit vector * weight)
    diff = dist[iu]
    unit = diff / np.where(pd > 0, pd, 1.0)            # (c_i - c_j)/d_ij
    w = p_w * unit
    grad = np.zeros((N, 3))
    np.add.at(grad, iu[0], w)
    np.add.at(grad, iu[1], -w)

    # wall contribution: d/d c_ik (0.5 - |c_ik|) = -sign(c_ik)
    gwall = -np.sign(c) * w_w.reshape(N, 3)
    grad += gwall
    return r_soft, grad, dist


def _refine(c0, betas, maxiter, step, deadline):
    """Annealed vectorized gradient ascent. Returns (best_c, best_true_r)."""
    c = np.asarray(c0, dtype=float).copy()
    best_c = _project(c)
    best_r = _true_radius(best_c)
    for beta in betas:
        for _ in range(maxiter):
            if time.time() > deadline - MARGIN:
                break
            r_soft, grad, _ = _soft_and_grad(c, beta)
            gnorm = np.linalg.norm(grad)
            if not np.isfinite(gnorm) or gnorm < 1e-13:
                break
            c = c + step * (grad / gnorm)
            c = _project(c)
            r = _true_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        if time.time() > deadline - MARGIN:
            break
    return best_c, best_r


def _scipy_polish(c0, deadline):
    """High-precision polish: maximize -soft radius with L-BFGS-B (tight tol)."""
    c = np.asarray(c0, dtype=float).copy()

    def neg_obj(x):
        r, _, _ = _soft_and_grad(x.reshape(N, 3), 2.5e4)
        return -r

    def neg_grad(x):
        _, g, _ = _soft_and_grad(x.reshape(N, 3), 2.5e4)
        return -g.ravel()

    try:
        res = minimize(
            neg_obj, c.ravel(), jac=neg_grad, method="L-BFGS-B",
            bounds=[(-0.5, 0.5)] * (3 * N),
            options={"maxiter": 800, "ftol": 1e-16, "gtol": 1e-10,
                     "maxls": 30, "maxfun": 4000},
        )
        c2 = _project(res.x.reshape(N, 3))
        if np.all(np.isfinite(c2)):
            return c2
    except Exception:
        pass
    return c


def _initial_grid():
    """Cubic lattice initialisation (deterministic), N points inside the cube."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(RNG_SEED)

    # ---- Stage A: annealed vectorized gradient ascent with restarts ----
    coarse_betas = [2.0, 5.0, 12.0, 30.0, 80.0]
    best_c = None
    best_r = -np.inf

    for restart in range(6):
        if time.time() > deadline - 30.0:      # keep ample time for polish
            break
        if restart == 0:
            c0 = _initial_grid()
        else:
            jit = 0.05 * (1.0 - 0.1 * restart)
            c0 = _project(_initial_grid() + jit * rng.standard_normal((N, 3)))
        c, r = _refine(c0, coarse_betas, 300, 0.05, deadline)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # ---- Stage B: fine anneal from the best coarse result ----
    if best_c is not None and time.time() < deadline - 25.0:
        c, r = _refine(best_c, [120.0, 200.0, 350.0], 200, 0.02, deadline)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # ---- Stage C: high-precision L-BFGS-B polish (tight tolerance) ----
    if best_c is not None and time.time() < deadline - 12.0:
        c = _scipy_polish(best_c, deadline)
        r = _true_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # ---- Final safety projection (guarantees a valid, in-cube result) ----
    if best_c is None:
        best_c = _initial_grid()
    out = _project(best_c)
    out = np.where(np.isfinite(out), out, 0.0)
    return out


# EVOLVE-BLOCK-END
