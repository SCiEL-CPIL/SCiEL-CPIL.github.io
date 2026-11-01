"""Rewrite: minimize log energy of 100 points on the unit sphere."""

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    t_start = time.time()
    DEADLINE = t_start + 115.0

    def fib_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = math.pi * (1.0 + math.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, 1e-15)
        return x / n

    def energy(x):
        # sum_{i<j} log(1/|xi-xj|)
        d = x[:, None, :] - x[None, :, :]
        dsq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dsq, 1.0)
        idx = np.triu_indices(N, 1)
        dsq = dsq[idx]
        return -np.sum(np.log(np.sqrt(dsq)))

    def energy_grad(x):
        x = np.asarray(x, dtype=float)
        d = x[:, None, :] - x[None, :, :]
        dsq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dsq, 1.0)
        dist = np.sqrt(dsq)

        idx = np.triu_indices(N, 1)
        en = -np.sum(np.log(dist[idx]))

        # gradient: for each i, sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # Use vectorized: W[i,j] = 1/|xi-xj|^2 for i != j
        W = np.zeros_like(dsq)
        W[idx] = 1.0 / dsq[idx]
        W = W + W.T

        with np.errstate(divide='ignore', invalid='ignore'):
            grad = np.sum(W[:, :, None] * d, axis=1)
        return en, grad

    def project(x):
        return norm_rows(x)

    # Phase 0: Fibonacci start
    best_pts = fib_sphere()
    best_e, _ = energy_grad(best_pts)

    # Riemannian gradient descent with backtracking
    def riemann_gd(x0, max_iter=8000, lr=0.01, lr_min=1e-7, tol=1e-12):
        x = project(x0.copy())
        best_x = x.copy()
        best_e, _ = energy_grad(x)
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            e, g = energy_grad(x)
            # Riemannian gradient (tangent projection)
            gx = np.sum(g * x, axis=1, keepdims=True)
            gt = g - gx * x

            gnorm = np.linalg.norm(gt, ord=np.inf)
            if gnorm < tol:
                break

            # Backtracking line search with Riemannian step
            step = lr
            for _ in range(30):
                new_x = project(x + step * gt)
                new_e, _ = energy_grad(new_x)
                if new_e < e:
                    break
                step *= 0.5

            x = new_x
            if new_e < best_e:
                best_e, _ = energy_grad(x)
                best_x = x.copy()
            if new_e < e - 1e-12:
                lr = min(lr * 1.05, 0.08)
            else:
                lr = max(lr * 0.85, lr_min)

        e_final, _ = energy_grad(best_x)
        return best_x, e_final

    # L-BFGS polish in ambient space (with projection afterwards)
    def lbfgs(x0):
        if time.time() > DEADLINE - 20:
            return x0, energy(x0)

        def obj_and_grad(v):
            x = v.reshape(-1, 3)
            e, g = energy_grad(x)
            return e, g.ravel()

        res = minimize(obj_and_grad, x0.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 600})
        xr = project(res.x.reshape(-1, 3))
        e, _ = energy_grad(xr)
        return xr, e

    # Run main optimization from Fibonacci
    if time.time() < DEADLINE - 40:
        pts, e = riemann_gd(best_pts, max_iter=5000, lr=0.01)
        if e < best_e:
            best_pts, best_e = pts, e

    if time.time() < DEADLINE - 20:
        pts, e = lbfgs(best_pts)
        if e < best_e:
            best_pts, best_e = pts, e

    # Second polish pass for extra precision
    if time.time() < DEADLINE - 10:
        pts, e = lbfgs(best_pts)
        if e < best_e:
            best_pts, best_e = pts, e

    # Random restarts with short Riemannian CG + L-BFGS
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 10
    while attempts < max_attempts and time.time() < DEADLINE - 18:
        # Mix random and perturbed
        if attempts % 3 == 0:
            x0 = project(rng.normal(size=(N, 3)))
        else:
            x0 = project(best_pts + 0.1 * rng.normal(size=(N, 3)))
        x0 = project(x0)

        # Short gradient descent
        if time.time() < DEADLINE - 25:
            p, e = riemann_gd(x0, max_iter=1500, lr=0.005)
            # Quick polish
            if time.time() < DEADLINE - 15:
                p, e2 = lbfgs(p)
                if e2 < best_e:
                    best_pts, best_e = p, e2
        attempts += 1

    # Final validation
    best_pts = project(best_pts)
    if not np.all(np.isfinite(best_pts)) or np.any(np.linalg.norm(best_pts, axis=1) < 1e-9):
        best_pts = fib_sphere()

    # Ensure no coincident points
    d = best_pts[:, None, :] - best_pts[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(dsq, 1.0)
    if np.min(dsq) < 1e-18:
        # perturb
        best_pts += 1e-8 * rng.normal(size=(N, 3))
        best_pts = project(best_pts)

    return best_pts
