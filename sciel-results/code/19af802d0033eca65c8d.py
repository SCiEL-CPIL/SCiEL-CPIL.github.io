# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective(points_flat, tau=20.0):
    """
    Smooth objective function to minimize.
    Minimizes 2 * (log(soft_max) - log(soft_min))
    which is -2 * log(R).
    """
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    
    # Avoid log(0) and numerical issues
    d = np.maximum(d, 1e-10)
    
    # Soft max: (1/tau) * log(sum(exp(tau * d)))
    # Soft min: -(1/tau) * log(sum(exp(-tau * d)))
    
    # Numerical stability for log-sum-exp
    max_d = np.max(d)
    min_d = np.min(d)
    
    # log(sum(exp(tau * d))) = tau * max_d + log(sum(exp(tau * (d - max_d))))
    log_sum_exp_max = tau * max_d + np.log(np.sum(np.exp(tau * (d - max_d))))
    
    # log(sum(exp(-tau * d))) = -tau * min_d + log(sum(exp(-tau * (d - min_d))))
    log_sum_exp_min = -tau * min_d + np.log(np.sum(np.exp(-tau * (d - min_d))))
    
    log_soft_max = log_sum_exp_max / tau
    log_soft_min = -log_sum_exp_min / tau
    
    # We want to minimize log(d_max) - log(d_min)
    # So objective = 2 * (log_soft_max - log_soft_min)
    val = 2.0 * (log_soft_max - log_soft_min)
    
    return val


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2

    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit

    best_points = None
    best_score = -1.0

    # Initial candidates
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])

    # 2. Random
    np.random.seed(42)
    p2 = np.random.randn(n, d)

    # 3. Grid-like (4x4)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    p3 = np.column_stack([xx.flatten(), yy.flatten()])

    # 4. Hexagonal packing (approximate)
    # 4 rows of 4 points, staggered
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x_coord = j * 1.0 + (0.5 if i % 2 else 0.0)
            y_coord = i * 0.866
            hex_pts.append([x_coord, y_coord])
    p4 = np.array(hex_pts)
    # Center and scale
    p4 = p4 - np.mean(p4, axis=0)
    
    # 5. Random with larger spread
    np.random.seed(123)
    p5 = np.random.randn(n, d) * 2.0

    candidates = [p1, p2, p3, p4, p5]
    
    # Add more random starts
    for i in range(20):
        np.random.seed(1000 + i)
        candidates.append(np.random.randn(n, d))

    tau = 20.0

    for i, init_pts in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        x0 = init_pts.flatten()

        # Use L-BFGS-B
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                args=(tau,),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )

            if res.success or res.status in [0, 1, 2, 3, 4]:
                pts_opt = res.x.reshape(n, 2)
                score = _compute_score(pts_opt)

                if score > best_score:
                    best_score = score
                    best_points = pts_opt

        except Exception:
            continue

    # If we have time, do a second pass with higher tau or more iterations on the best point
    if best_points is not None and time.time() < deadline - 60:
        tau2 = 50.0
        x0 = best_points.flatten()
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                args=(tau2,),
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.success:
                pts_opt = res.x.reshape(n, 2)
                score = _compute_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_points = pts_opt
        except Exception:
            pass

    if best_points is None:
        # Fallback to seed
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points
# EVOLVE-BLOCK-END
