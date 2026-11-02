"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 140  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area.
    Uses vectorized operations for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # dx[i, j] = x[j] - x[i]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Calculate cross product terms for all i, j, k
    # term1[i, j, k] = dx[i, j] * dy[i, k]
    # term2[i, j, k] = dy[i, j] * dx[i, k]
    term1 = dx[:, :, None] * dy[:, None, :]
    term2 = dy[:, :, None] * dx[:, None, :]
    cross = term1 - term2
    areas = 0.5 * np.abs(cross)
    
    # Mask for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    if y < 0 or y > S3H:
        return False
    if x < 0 or x > 1:
        return False
    if y > S3H * (1 - x):
        return False
    if y > S3H * x:
        return False
    return True


def _penalized_min_area(x):
    """Objective for optimization: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        # Distances to boundaries (positive if outside)
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    return -min_a + 10.0 * penalty


def construct():
    start_time = time.time()
    
    # Initial random configuration
    rng = np.random.RandomState(42)
    
    def random_point_in_triangle():
        # Rejection sampling
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                # Ensure strictly inside to avoid boundary issues in initial guess
                if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                    if y < S3H * (1 - x) * 0.99 and y < S3H * x * 0.99:
                        return [x, y]

    best_points = None
    best_score = -float('inf')
    
    n_restarts = 50  # More restarts with faster objective
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Initial guess
        x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _penalized_min_area, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
        except Exception:
            continue
        
        if time.time() - start_time > DEADLINE - 5:
            break
            
        pts = res.x.reshape(-1, 2)
        
        # Check validity and score
        # The penalty should have kept them inside, but verify
        valid = True
        for p in pts:
            if not _is_inside_triangle(p):
                valid = False
                break
        
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return a valid random configuration
        best_points = np.array([random_point_in_triangle() for _ in range(N)])
    
    # Final polish on the best candidate if time permits
    if time.time() - start_time < DEADLINE - 2:
        try:
            res = minimize(
                _penalized_min_area, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished = res.x.reshape(-1, 2)
            valid = all(_is_inside_triangle(p) for p in polished)
            if valid:
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_points = polished
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
