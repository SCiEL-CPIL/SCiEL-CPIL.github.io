"""Optimize 100 points on the unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 110.0  # safety margin before the 120 s kill
EPS = 1e-12


def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3} (Euclidean, not yet
    projected to the tangent space).
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask the diagonal and avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    # Use a floor to avoid log(0); the diagonal is masked anyway
    dist_safe = np.where(mask, np.maximum(dist, EPS), 1.0)
    log_term = np.log(1.0 / dist_safe)  # = -log(dist_safe)
    energy = np.sum(log_term[mask])

    # Gradient: d/dp_i sum_{j!=i} log(1/|p_i - p_j|)
    # d/dp_i log(1/|p_i - p_j|) = - (p_i - p_j) / |p_i - p_j|^2
    # = (p_j - p_i) / |p_i - p_j|^2
    # We accumulate for each i the sum over j of (p_j - p_i)/dist^2
    # dist^2 matrix:
    dist2 = dist * dist
    dist2_safe = np.where(mask, np.maximum(dist2, EPS), 1.0)
    inv_dist2 = np.where(mask, 1.0 / dist2_safe, 0.0)
    # For each i, sum over j of (p_j - p_i) * inv_dist2[i, j]
    # (p_j - p_i) for fixed i and all j: pts[None, :, :] - pts[:, None, :] -> (N, N, 3)
    # Actually diff = p_i - p_j, so p_j - p_i = -diff
    grad = -np.sum(diff * inv_dist2[:, :, None], axis=1)
    return energy, grad


def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space of the unit sphere."""
    # Tangent projection: g_t = g - (g . p) p
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts


def _energy_tangent(pts):
    """Energy only, for use in a line search if needed."""
    e, _ = _energy_and_grad(pts)
    return e


def construct():
    t_start = time.time()

    # --- Initial configuration: Fibonacci spiral ---
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Ensure unit length
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    best_pts = pts.copy()
    best_energy = _energy_tangent(pts)

    # L-BFGS-B on the sphere using tangent-space gradient.
    # We optimize in the full R^{N*3} space but project the gradient.
    # The constraint ||p_i|| = 1 is handled by the projection; however,
    # L-BFGS-B doesn't know about the manifold, so we use a trick:
    # we define the objective on the tangent space by treating the
    # gradient as already projected and letting the optimizer move in
    # R^{N*3}, then re-normalize. This is not strictly manifold-optimized
    # but works well in practice for this problem.

    # Alternative: use a simple gradient descent with line search on the
    # tangent space, which is more principled. Let's do a hybrid:
    # 1) A few steps of gradient descent with backtracking line search.
    # 2) Then L-BFGS-B on the full coordinates with the projected gradient.

    # Actually, let's use scipy.optimize.minimize with method='L-BFGS-B'
    # and provide the projected gradient. The optimizer will move in
    # R^{N*3} and we re-normalize after each step. To do this cleanly,
    # we can use a wrapper that re-normalizes.

    # Simpler approach: use the projected gradient in a manual gradient
    # descent loop with a decreasing step size, then a final L-BFGS-B.

    # Let's do a manual gradient descent with line search first.
    x = pts.ravel().copy()
    lr = 0.01
    n_iter = 2000

    for it in range(n_iter):
        if time.time() - t_start > DEADLINE_SECONDS:
            break
        x_pts = x.reshape(N, 3)
        # Re-normalize
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        g_flat = g_t.ravel()

        # Backtracking line search
        alpha = lr
        e_new = e
        for _ in range(20):
            x_trial = x_pts.ravel() + alpha * g_flat
            x_trial_pts = x_trial.reshape(N, 3)
            x_trial_pts = x_trial_pts / np.linalg.norm(x_trial_pts, axis=1, keepdims=True)
            e_trial = _energy_tangent(x_trial_pts)
            if e_trial < e_new:
                x = x_trial
                x_pts = x_trial_pts
                e_new = e_trial
                break
            alpha *= 0.5
        else:
            # No improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
            continue

        # Track best
        if e_new < best_energy:
            best_energy = e_new
            best_pts = x_pts.copy()

        # Decrease learning rate slowly
        lr = max(0.005, lr * 0.999)

    # Final L-BFGS-B polish on the full coordinates
    if time.time() - t_start < DEADLINE_SECONDS - 5:
        x0 = best_pts.ravel()

        def fun_and_grad(x):
            x_pts = x.reshape(N, 3)
            x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
            e, g = _energy_and_grad(x_pts)
            g_t = _tangent_grad(x_pts, g)
            return e, g_t.ravel()

        try:
            res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 200, 'gtol': 1e-8,
                                    'ftol': 1e-12, 'maxfun': 500})
            if res.success or res.fun < best_energy:
                x_opt = res.x.reshape(N, 3)
                x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
                e_opt = _energy_tangent(x_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = x_opt
        except Exception:
            pass

    # Final check: ensure all points are on the unit sphere and distinct
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
