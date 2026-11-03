"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.special as sps
from scipy.optimize import minimize

# ---------- geometry helpers ----------

def _project(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return pts / n


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    return _project(p)


def _sobol_like(n, seed):
    """Low-discrepancy-ish starting points via Halton-like construction."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    i = (np.arange(n) + 0.5) / n
    # Halton base-3 for x, base-2 for y
    def halton(k, base):
        f, r = 1.0, 0.0
        while k:
            f /= base
            r += f * (k % base)
            k //= base
        return r
    xs = np.array([halton(int(perm[k]) + 1, 3) for k in range(n)])
    ys = np.array([halton(int(perm[k]) + 1, 2) for k in range(n)])
    theta = 2 * np.pi * xs
    cosphi = 1 - 2 * ys
    sinphi = np.sqrt(np.maximum(1 - cosphi * cosphi, 0.0))
    return np.column_stack([sinphi * np.cos(theta), sinphi * np.sin(theta), cosphi])


# ---------- energy & gradient (vectorized) ----------

def _energy(pts):
    """Fast energy using pairwise dot products (vectorized)."""
    pts = np.asarray(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)
    d2 = 2.0 - dots
    d2 = np.maximum(d2, 1e-30)
    dist = np.sqrt(d2)
    iu = np.triu_indices(len(pts), 1)
    return -np.sum(np.log(dist[iu]))


def _energy_grad(pts):
    """Vectorized energy and gradient on the sphere."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)
    d2 = 2.0 - dots
    d2 = np.maximum(d2, 1e-30)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(dist[iu]))

    # grad_i = -sum_{j!=i} (x_i - x_j)/dist_ij^2 = sum_{j!=i} (x_j - x_i)/d2_ij
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[iu] = 1.0 / d2[iu]
    inv_d2 = 0.5 * (inv_d2 + inv_d2.T)  # symmetric
    # grad_i = sum_j inv_d2_ij * (x_j - x_i) = (X^T W)_i - (sum_j W_ij) x_i
    grad = (inv_d2 @ pts) - (inv_d2.sum(axis=1, keepdims=True) * pts)
    return energy, grad


def _precise_energy(pts):
    """Exact energy using math.fsum over pair terms (matches evaluator)."""
    pts = np.asarray(pts)
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            d = max(d, 1e-15)
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


# ---------- Riemannian L-BFGS on the sphere ----------

def _riemannian_lbfgs(pts, maxiter=400, ftol=1e-14, gtol=1e-11, maxls=50):
    """L-BFGS with retraction to the sphere after each step."""
    x = pts.copy()
    m = 20
    W = []
    for _ in range(maxiter):
        f, g = _energy_grad(x)
        gn = np.linalg.norm(g)
        if gn < gtol:
            break
        # L-BFGS two-loop recursion
        q = g.copy()
        r = np.zeros_like(q)
        a = []
        for (w, y, gg) in reversed(W):
            gamma = float(np.dot(w, y))
            alpha = float(np.dot(w, q)) / gamma
            r = r + alpha * y
            a.append(alpha)
        r = r / (float(np.dot(y, q)) / gamma)
        for (w, y, gg), alpha in zip(W, reversed(a)):
            beta = float(np.dot(y, r)) / float(np.dot(y, y))
            q = q + alpha * y - beta * r
        d = -q
        # line search (Armijo backtracking) with retraction
        c1 = 1e-4
        step = 1.0
        accepted = False
        for _ in range(maxls):
            x_trial = _project(x + step * d)
            f_trial, _ = _energy_grad(x_trial)
            if f_trial <= f + c1 * step * float(np.dot(d, g)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        # update L-BFGS memory
        y = _energy_grad(x_trial)[1] - g
        wy = float(np.dot(w, y))
        if wy > 1e-16:
            W.append((d, y, wy))
            if len(W) > m:
                W.pop(0)
        x = x_trial
    return x, _energy(x)


# ---------- simple Riemannian gradient descent (fallback / warm-start) ----------

def _grad_desc(pts, steps=300, lr0=0.02, lr_min=1e-4):
    x = pts.copy()
    lr = lr0
    for _ in range(steps):
        f, g = _energy_grad(x)
        x = _project(x - lr * g)
        lr = max(lr * 0.99, lr_min)
    return x, _energy(x)


def construct():
    start = time.time()
    deadline = start + 138.0  # safety margin under 150s

    def remaining():
        return deadline - time.time()

    # ---- Phase 1: L-BFGS from a good structured start ----
    best_pts = _fibonacci_spiral(N).copy()
    best_energy = _precise_energy(best_pts)

    def try_improve(candidate):
        nonlocal best_pts, best_energy
        candidate = _project(candidate)
        e = _precise_energy(candidate)
        if e < best_energy:
            best_pts = candidate.copy()
            best_energy = e

    try:
        x, _ = _riemannian_lbfgs(best_pts, maxiter=500, ftol=1e-14, gtol=1e-11)
        try_improve(x)
    except Exception:
        pass

    if remaining() < 30.0:
        return best_pts

    # ---- Phase 2: diverse restarts (random + low-discrepancy), each refined ----
    rng = np.random.default_rng(12345)
    starts = []
    # random restarts
    for s in range(14):
        starts.append(_random_sphere(N, seed=1000 + s))
    # low-discrepancy restarts
    for s in range(6):
        starts.append(_sobol_like(N, seed=7000 + s))

    for idx, st in enumerate(starts):
        if remaining() < 18.0:
            break
        # warm-start with a few gradient steps, then L-BFGS
        try:
            x, _ = _grad_desc(st, steps=40, lr0=0.02)
            x, _ = _riemannian_lbfgs(x, maxiter=300, ftol=1e-13, gtol=1e-11)
            try_improve(x)
        except Exception:
            pass

    if remaining() < 12.0:
        return best_pts

    # ---- Phase 3: ultra-precise polish on the best configuration ----
    try:
        x, _ = _riemannian_lbfgs(best_pts, maxiter=1500, ftol=1e-16, gtol=1e-13)
        try_improve(x)
    except Exception:
        pass

    # tiny perturbations to escape any shallow local minima
    for _ in range(6):
        if remaining() < 4.0:
            break
        try:
            pert = _project(best_pts + rng.standard_normal((N, 3)) * 1e-5)
            x, _ = _riemannian_lbfgs(pert, maxiter=400, ftol=1e-15, gtol=1e-12)
            try_improve(x)
        except Exception:
            pass

    best_pts = _project(best_pts)

    # ---- final validation: no two points closer than 1e-9, finite values ----
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-9:
        # fall back to well-separated fibonacci spiral
        best_pts = _fibonacci_spiral(N)

    return best_pts

# EVOLVE-BLOCK-END
