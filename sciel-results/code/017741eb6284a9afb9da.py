# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    n = 16
    d = 2

    def compute_score(points):
        if points.shape != (n, d):
            return -np.inf
        dists = pdist(points)
        if dists.size == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(points):
        """Smooth surrogate: log-sum-exp based soft-min/soft-max"""
        dists = pdist(points)
        if dists.size == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        # Use log-sum-exp for smooth min/max
        beta = 50.0
        log_min = -1.0 / beta * np.log(np.sum(np.exp(-beta * dists)))
        log_max = 1.0 / beta * np.log(np.sum(np.exp(beta * dists)))
        soft_ratio = np.exp(log_min - log_max)
        return soft_ratio * soft_ratio

    def gradient_descent(points, lr=0.01, steps=1000, tol=1e-10):
        """Gradient descent with numerical gradients"""
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            # Compute gradient numerically
            grad = np.zeros_like(points)
            eps = 1e-6
            base_obj = objective(points)
            for i in range(n):
                for j in range(d):
                    points[i, j] += eps
                    obj_plus = objective(points)
                    points[i, j] -= 2 * eps
                    obj_minus = objective(points)
                    points[i, j] += eps
                    grad[i, j] = (obj_plus - obj_minus) / (2 * eps)
            # Update
            points += lr * grad
            # Normalize to prevent drift
            points -= np.mean(points, axis=0)
            scale = np.max(np.abs(points))
            if scale > 0:
                points /= scale
            if step % 100 == 0:
                pass  # Could check convergence

        return points

    def simulated_annealing(points, iterations=5000, t_start=1.0, t_end=0.001):
        """Simulated annealing to escape local optima"""
        current = points.copy()
        current_score = compute_score(current)
        best = current.copy()
        best_score = current_score

        for it in range(iterations):
            if time.time() > deadline - 5.0:
                break
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (it / iterations)

            # Random perturbation
            step_size = t * 0.1
            perturbed = current + np.random.randn(n, d) * step_size
            perturbed -= np.mean(perturbed, axis=0)
            scale = np.max(np.abs(perturbed))
            if scale > 0:
                perturbed /= scale

            perturbed_score = compute_score(perturbed)

            # Acceptance criterion
            if perturbed_score > current_score or np.random.random() < np.exp((perturbed_score - current_score) / max(t, 1e-10)):
                current = perturbed
                current_score = perturbed_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score

        return best, best_score

    # Initialize with multiple structured configurations
    initializations = []

    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(init1)

    # 2. Two concentric circles (8 points each)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    outer = np.column_stack([np.cos(angles_outer) * 1.0, np.sin(angles_outer) * 1.0])
    inner = np.column_stack([np.cos(angles_inner) * 0.5, np.sin(angles_inner) * 0.5])
    init2 = np.vstack([outer, inner])
    initializations.append(init2)

    # 3. Grid-like arrangement (4x4)
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    init3 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    initializations.append(init3)

    # 4. Random with seed
    np.random.seed(42)
    init4 = np.random.randn(n, d)
    init4 -= np.mean(init4, axis=0)
    scale = np.max(np.abs(init4))
    if scale > 0:
        init4 /= scale
    initializations.append(init4)

    # Optimize each initialization
    best_points = None
    best_score = -np.inf

    for init in initializations:
        if time.time() > deadline - 30.0:
            break

        # Gradient descent first
        gd_points = gradient_descent(init.copy(), lr=0.01, steps=500)
        gd_score = compute_score(gd_points)

        if gd_score > best_score:
            best_score = gd_score
            best_points = gd_points.copy()

        # Simulated annealing from GD result
        sa_points, sa_score = simulated_annealing(gd_points, iterations=3000, t_start=0.5, t_end=0.001)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()

    # Final polish with tighter tolerance
    if best_points is not None and time.time() < deadline - 10.0:
        polished = gradient_descent(best_points.copy(), lr=0.005, steps=2000, tol=1e-12)
        polished_score = compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score

    # Ensure valid output
    if best_points is None:
        best_points = initializations[0]

    # Final normalization
    best_points -= np.mean(best_points, axis=0)
    scale = np.max(np.abs(best_points))
    if scale > 0:
        best_points /= scale

    return best_points


# EVOLVE-BLOCK-END
