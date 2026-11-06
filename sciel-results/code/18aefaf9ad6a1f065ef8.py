"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
# Deadline guard: 150 s wall-clock limit, keep a 15 s safety margin
_DEADLINE = time.time() + 135.0

# Precompute upper-triangle indices for the pair sum
_IU = np.triu_indices(N, k=1)


def _retract(pts):
    """Project points back to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _pair_distances(pts):
    """150x150 distance matrix via d^2 = 2 - 2*(x_i . x_j)."""
    gram = pts @ pts.T
    d2 = 2.0 - 2.0 * gram
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 1.0)
    return np.sqrt(d2)


def _energy(pts):
    """Vectorized sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    dist = _pair_distances(pts)
    dist = np.maximum(dist, 1e-12)
    e = -np.log(dist)
    terms = e[_IU[0], _IU[1]].tolist()
    return math.fsum(terms)


def _grad(pts):
    """Vectorized gradient of E wrt each point, tangent-projected."""
    dist = _pair_distances(pts)
    dist = np.maximum(dist, 1e-12)
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    row_sums = inv_d2.sum(axis=1)
    grad = -pts * row_sums[:, np.newaxis] + inv_d2 @ pts
    dot = np.einsum('ij,ij->i', grad, pts)
    grad = grad - dot[:, np.newaxis] * pts
    return grad


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    return _retract(pts)


def _obj_grad(x_flat):
    x = _retract(x_flat.reshape(N, 3))
    e = _energy(x)
    g = _grad(x)
    return e, g.flatten()


def _lbfgs_optimize(pts, max_iter=4000, tol=1e-10, maxcor=20):
    """Optimize via L-BFGS-B on the sphere (ambient + retract)."""
    pts = _retract(pts.copy())
    best_pts = pts.copy()
    best_E = _energy(pts)

    def f(x):
        return _obj_grad(x)[0]

    def j(x):
        return _obj_grad(x)[1]

    result = minimize(
        f, pts.flatten(), jac=j, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                 'maxfun': max_iter, 'maxls': 50, 'maxcor': maxcor}
    )
    if result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    return best_pts, best_E


def _gd_polish(pts, max_iter=3000, lr_init=1e-5, tol=1e-11, time_limit=None):
    """Riemannian gradient descent with backtracking line search."""
    if time_limit is None:
        time_limit = _DEADLINE - 2.0
    pts = _retract(pts.copy())
    best_pts = pts.copy()
    best_e = _energy(pts)
    lr = lr_init
    c1 = 1e-4

    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        g = _grad(pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        e_curr = _energy(pts)
        slope = -np.sum(g * g)
        step = 1.0
        min_step = 1e-12
        found = False
        while step >= min_step:
            new_pts = _retract(pts - lr * step * g)
            e_new = _energy(new_pts)
            if not np.isfinite(e_new):
                step *= 0.5
                continue
            if e_new <= e_curr + c1 * (lr * step) * slope:
                found = True
                break
            step *= 0.5
        if not found:
            lr *= 0.5
            if lr < 1e-16:
                break
            continue
        pts = new_pts
        e_new = _energy(pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = pts.copy()
        if step > 0.9:
            lr = min(lr * 1.1, 1e-3)
        else:
            lr = max(lr * 0.9, 1e-16)
    return best_pts, best_e


def construct():
    t0 = time.time()

    # 1. Fibonacci start + L-BFGS
    pts0 = _fibonacci_sphere(N)
    best_pts, best_E = _lbfgs_optimize(pts0, max_iter=4000, tol=1e-10)

    # 2. Random restarts
    seeds = [42, 123, 999, 2023, 777, 555, 314, 271, 161, 83]
    for seed in seeds:
        if time.time() > _DEADLINE - 35:
            break
        p = _random_sphere(N, seed)
        p2, E2 = _lbfgs_optimize(p, max_iter=3000, tol=1e-10, maxcor=15)
        if E2 < best_E:
            best_pts, best_E = p2, E2

    # 3. Jittered Fibonacci restarts
    if time.time() < _DEADLINE - 30:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() > _DEADLINE - 25:
                break
            p = _fibonacci_sphere(N)
            p += rng.normal(0, 0.1, (N, 3))
            p = _retract(p)
            p2, E2 = _lbfgs_optimize(p, max_iter=3000, tol=1e-10, maxcor=15)
            if E2 < best_E:
                best_pts, best_E = p2, E2

    # 4. Final high-precision Riemannian GD polish
    if time.time() < _DEADLINE - 8:
        polished_pts, polished_E = _gd_polish(
            best_pts, max_iter=5000, lr_init=1e-6, tol=1e-12,
            time_limit=_DEADLINE - 2.0
        )
        if polished_E < best_E:
            best_pts, best_E = polished_pts, polished_E

    # Final normalization
    best_pts = _retract(best_pts)
    return best_pts
