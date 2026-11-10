"""Optimized: gradient descent on sphere for log-energy minimization."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy(pts):
    """Compute log energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _log_energy_and_grad(pts):
    """Compute log energy and its gradient.
    E = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)

    # Energy: sum over upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))

    # Gradient: for each i, sum over j of (x_i - x_j)/d_ij^2
    # weight[i,j] = 1/d_ij^2 for i != j
    weight = 1.0 / dist2  # (n, n)
    np.fill_diagonal(weight, 0.0)
    # grad[i] = sum_j weight[i,j] * (x_i - x_j)
    # = sum_j weight[i,j]*x_i - sum_j weight[i,j]*x_j
    # = x_i * sum_j weight[i,j] - (weight @ x)[i]
    row_sum = weight.sum(axis=1)  # (n,)
    wx = weight @ pts  # (n, 3)
    grad = pts * row_sum[:, np.newaxis] - wx  # (n, 3)

    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _gradient_descent(pts, time_budget, min_step=1e-8, max_step=0.1):
    """Gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    step = 0.01
    patience = 0
    max_patience = 500

    while time.time() < time_budget:
        energy, grad = _log_energy_and_grad(pts)

        # Project gradient to tangent space: grad_tan = grad - (grad.x) x
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tan = grad - dot * pts

        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-12:
            break

        # Backtracking line search
        alpha = step
        improved = False
        for _ in range(20):
            new_pts = pts + alpha * grad_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy = _log_energy(new_pts)

            if new_energy < current_energy - 1e-15:
                improved = True
                break
            alpha *= 0.5

        if improved:
            pts = new_pts
            current_energy = new_energy
            patience = 0
            # Increase step slightly if we made good progress
            step = min(step * 1.05, max_step)
        else:
            step *= 0.5
            patience += 1
            if step < min_step or patience > max_patience:
                break

        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = pts.copy()

    return _project_to_sphere(best_pts), best_energy


def _restart_optimize(start_pts, time_budget):
    """Run gradient descent from a given start."""
    return _gradient_descent(start_pts, time_budget)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Multiple restarts with different initializations
    n_restarts = 8
    per_restart_budget = (deadline - start_time) / n_restarts * 0.6

    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Generate diverse starting points
        if r == 0:
            start = _fibonacci_spiral(N)
        elif r == 1:
            # Slightly perturbed fibonacci
            start = _fibonacci_spiral(N)
            start += np.random.normal(0, 0.01, start.shape)
            start = _project_to_sphere(start)
        elif r == 2:
            # Random start
            start = np.random.randn(N, 3)
            start = _project_to_sphere(start)
        elif r == 3:
            # Fibonacci with different offset
            i = np.arange(N) + 0.3
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5
            rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            start = np.column_stack([rr * np.cos(phi), rr * np.sin(phi), z])
        elif r == 4:
            # Fibonacci with different offset 2
            i = np.arange(N) + 0.7
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 1.0
            rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            start = np.column_stack([rr * np.cos(phi), rr * np.sin(phi), z])
        elif r == 5:
            # Random with some structure
            theta = np.random.uniform(0, np.pi, N)
            phi = np.random.uniform(0, 2 * np.pi, N)
            start = np.column_stack([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
        elif r == 6:
            # Two-layer structure
            half = N // 2
            top = _fibonacci_spiral(half)
            bot = _fibonacci_spiral(N - half)
            start = np.vstack([top, bot * 0.5])  # compress bottom
            start = _project_to_sphere(start)
        else:
            # More random
            start = np.random.randn(N, 3)
            start = _project_to_sphere(start)

        restart_budget = min(deadline - 10, time.time() + per_restart_budget)
        pts, energy = _restart_optimize(start, restart_budget)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

    # Phase 2: Fine-tune the best result with more iterations
    if best_pts is not None and time.time() < deadline - 5:
        fine_budget = deadline - 5
        pts, energy = _gradient_descent(best_pts, fine_budget, min_step=1e-10, max_step=0.05)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

    # Final validation
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    best_pts = _project_to_sphere(best_pts)

    # Validate
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    if best_pts.shape != (N, 3):
        best_pts = _fibonacci_spiral(N)

    return best_pts
# EVOLVE-BLOCK-END
