# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin before 630s kill

    n = 16

    # Initial guess: 4x4 square grid, centered
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    points_init = np.column_stack((X.ravel(), Y.ravel()))
    points_init -= points_init.mean(axis=0)  # center

    def compute_score(pts):
        """Compute R = (d_min/d_max)^2"""
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = d.min()
        d_max = d.max()
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def negative_log_ratio(pts_flat):
        """
        Smooth surrogate: minimize -log(d_min) + log(d_max)
        which is equivalent to minimizing log(d_max/d_min) = -log(d_min/d_max)
        We use soft-min and soft-max for differentiability.
        """
        pts = pts_flat.reshape(-1, 2)
        d = pdist(pts)
        if len(d) == 0:
            return 0.0

        # Soft min: -log(sum(exp(-d/beta))) * beta approximates min(d)
        # Soft max: log(sum(exp(d/beta))) * beta approximates max(d)
        beta = 0.01
        d_clipped = np.clip(d, 1e-10, None)

        # Use log-sum-exp for numerical stability
        # soft_max = beta * log(sum(exp(d/beta)))
        d_scaled_max = d_clipped / beta
        max_val = d_scaled_max.max()
        soft_max = beta * (max_val + np.log(np.sum(np.exp(d_scaled_max - max_val))))

        # soft_min = -beta * log(sum(exp(-d/beta)))
        d_scaled_min = -d_clipped / beta
        min_val = d_scaled_min.max()
        soft_min = -beta * (min_val + np.log(np.sum(np.exp(d_scaled_min - min_val))))

        # We want to maximize d_min/d_max, so minimize log(d_max) - log(d_min)
        return soft_max - soft_min

    # Phase 1: Optimize from grid start
    best_points = points_init.copy()
    best_score = compute_score(best_points)

    x0 = points_init.ravel()

    try:
        result = minimize(
            negative_log_ratio,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000}
        )
        if result.success or result.fun < negative_log_ratio(x0):
            candidate = result.x.reshape(-1, 2)
            candidate -= candidate.mean(axis=0)
            score = compute_score(candidate)
            if score > best_score:
                best_points = candidate
                best_score = score
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 30:
        return best_points

    # Phase 2: Try multiple random restarts with different scales
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 60:
            break

        # Different initial configurations
        if restart == 0:
            # Slightly perturbed grid
            scale = 1.0 + 0.1 * restart
            x = np.linspace(-1.5*scale, 1.5*scale, 4)
            y = np.linspace(-1.5*scale, 1.5*scale, 4)
            X, Y = np.meshgrid(x, y)
            pts = np.column_stack((X.ravel(), Y.ravel()))
        else:
            # Random points in a disk
            angles = np.random.uniform(0, 2*np.pi, n)
            radii = np.random.uniform(0, 1, n) ** 0.5
            pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

        pts -= pts.mean(axis=0)

        if compute_score(pts) <= best_score:
            continue

        x0 = pts.ravel()
        try:
            result = minimize(
                negative_log_ratio,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 15000}
            )
            if result.success or result.fun < negative_log_ratio(x0):
                candidate = result.x.reshape(-1, 2)
                candidate -= candidate.mean(axis=0)
                score = compute_score(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
        except Exception:
            pass

    # Phase 3: Final polish with tighter tolerance
    if time.time() > deadline - 20:
        return best_points

    x0 = best_points.ravel()
    try:
        result = minimize(
            negative_log_ratio,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
        )
        if result.success or result.fun < negative_log_ratio(x0):
            candidate = result.x.reshape(-1, 2)
            candidate -= candidate.mean(axis=0)
            score = compute_score(candidate)
            if score > best_score:
                best_points = candidate
                best_score = score
    except Exception:
        pass

    return best_points


# EVOLVE-BLOCK-END
