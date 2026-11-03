"""Rewritten program for the 50-sphere packing in a unit cube task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50


def _radius(c):
    """Compute the maximum common radius for centres c (N,3)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Distance to walls
    d_wall = np.min(0.5 - np.abs(c), axis=1)
    r_wall = np.min(d_wall)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_wall, r_pair)


def _neg_radius(c):
    """Negative radius for minimization."""
    return -_radius(c)


def _grad_neg_radius(c):
    """Gradient of negative radius (approximate via finite differences if needed)."""
    # For simplicity and speed, use analytic where possible or fall back to finite diff
    # Here we use a simple finite difference gradient for robustness
    eps = 1e-6
    g = np.zeros_like(c)
    for i in range(c.size):
        old = c.flat[i]
        c.flat[i] = old + eps
        f1 = _neg_radius(c)
        c.flat[i] = old - eps
        f0 = _neg_radius(c)
        c.flat[i] = old
        g.flat[i] = (f1 - f0) / (2 * eps)
    return g


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial configuration: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    best_c = c.copy()
    best_r = _radius(best_c)

    # Local optimization using L-BFGS-B with bounds
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)

    # Try a few restarts with small random perturbations
    rng = np.random.default_rng(42)
    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break

        if restart == 0:
            c_curr = c.copy()
        else:
            # Add small random perturbation
            c_curr = best_c + rng.normal(0, 0.01, size=(N, 3))
            # Clip to bounds
            c_curr = np.clip(c_curr, -0.5 + 1e-9, 0.5 - 1e-9)

        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _neg_radius,
                c_curr.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6},
            )
            if result.success and result.fun < -best_r:
                best_r = -result.fun
                best_c = result.x.reshape(N, 3)
        except Exception:
            pass

        # Check deadline
        if time.time() > deadline - 10.0:
            break

    # Final polish: a few more iterations with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _neg_radius,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if result.success and result.fun < -best_r:
                best_r = -result.fun
                best_c = result.x.reshape(N, 3)
        except Exception:
            pass

    # Ensure validity: clip to cube
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_c
