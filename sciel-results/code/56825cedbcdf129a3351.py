# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    Uses a parameterized 14-gon with radial modulation and out-of-plane displacement,
    then refines with direct optimization on the 14 points.
    """
    start_time = time.time()
    DEADLINE = 600.0  # safety margin before 630s kill

    n = 14
    d = 3

    def metric(points):
        """Compute (d_min/d_max)^2, return 0 if invalid."""
        try:
            if not np.all(np.isfinite(points)):
                return 0.0
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0:
                return 0.0
            return (d_min / d_max) ** 2
        except Exception:
            return 0.0

    def neg_metric(points):
        return -metric(points)

    # Strategy 1: Parameterized 14-gon in 3D
    # Points on a circle in xy-plane with radial modulation and z-displacement
    def make_param_points(params):
        """params: [epsilon, z_amp, phase_z, rot_x, rot_y]"""
        eps, z_amp, phase_z, rot_x, rot_y = params
        theta = 2 * np.pi * np.arange(n) / n
        r = 1.0 + eps * np.cos(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_amp * np.sin(phase_z + theta)
        
        points = np.column_stack([x, y, z])
        
        # Apply small rotations to break any residual symmetry
        cx, sx = np.cos(rot_x), np.sin(rot_x)
        Ry = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        cy, sy = np.cos(rot_y), np.sin(rot_y)
        Rx = np.array([[1,0,0],[0,cy,-sy],[0,sy,cy]])
        
        points = points @ Rx.T @ Ry.T
        return points

    # Optimize the 5 parameters
    best_points = None
    best_score = 0.0

    # Try multiple initial guesses for the parameter optimization
    init_guesses = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.2, np.pi/4, 0.0, 0.0],
        [0.1, 0.3, np.pi/3, 0.1, 0.1],
        [0.15, 0.15, np.pi/6, 0.2, 0.1],
        [0.05, 0.2, 0.0, 0.3, 0.2],
        [0.2, 0.1, np.pi/2, 0.1, 0.3],
    ]

    for guess in init_guesses:
        if time.time() > DEADLINE - 30:
            break
        try:
            res = minimize(
                lambda p: -metric(make_param_points(p)),
                guess,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            pts = make_param_points(res.x)
            sc = metric(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass

    # Strategy 2: Direct optimization on 14 points (42 DOF)
    # Start from the best parameterized result or a random start
    if best_points is not None:
        start_pts = best_points.copy()
    else:
        np.random.seed(42)
        start_pts = np.random.randn(n, d)

    # Normalize to unit scale
    start_pts = start_pts / np.linalg.norm(start_pts)

    if time.time() < DEADLINE - 60:
        try:
            res = minimize(
                neg_metric,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = res.x.reshape(n, d)
            sc = metric(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        except Exception:
            pass

    # Strategy 3: Polish with multiple random restarts on the 42D space
    if best_points is not None and time.time() < DEADLINE - 30:
        for restart in range(5):
            if time.time() > DEADLINE - 30:
                break
            np.random.seed(100 + restart)
            # Perturb the best solution
            perturbed = best_points + 0.05 * np.random.randn(n, d)
            try:
                res = minimize(
                    neg_metric,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = res.x.reshape(n, d)
                sc = metric(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
            except Exception:
                pass

    # Final safety: if we have no valid points, fall back to random
    if best_points is None or not np.all(np.isfinite(best_points)):
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    # Ensure finite
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return best_points

# EVOLVE-BLOCK-END
