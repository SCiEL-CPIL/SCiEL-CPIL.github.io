"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 140  # safety margin from 150s limit


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a


def _is_inside_triangle(p):
    """Check if point p is inside or on the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    if y < -1e-10 or y > S3H + 1e-10:
        return False
    if x < -1e-10:
        return False
    if x > 1 + 1e-10:
        return False
    # Left boundary: y <= sqrt(3)*x
    if y > S3 * x + 1e-10:
        return False
    # Right boundary: y <= sqrt(3)*(1-x)
    if y > S3 * (1 - x) + 1e-10:
        return False
    return True


def _all_inside(pts):
    """Check if all points are inside the triangle."""
    for p in pts:
        if not _is_inside_triangle(p):
            return False
    return True


def _penalized_min_area(flat_params):
    """Objective: minimize negative of min area, with penalty for points outside triangle."""
    points = flat_params.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 100
        if y > S3H:
            penalty += (y - S3H) * 100
        if x < 0:
            penalty += abs(x) * 100
        if x > 1:
            penalty += (x - 1) * 100
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 100
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 100
    ma = _min_triangle_area(points) / TRI_AREA
    return -ma + penalty


def _random_point_in_triangle(rng):
    """Generate a random point uniformly in the triangle using barycentric coords."""
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    x = r1 + 0.5 * r2
    y = S3H * r2
    return np.array([x, y])


def _project_to_triangle(p):
    """Project point p to the closest point on/inside the triangle."""
    x, y = p
    # Clip y to [0, S3H]
    y = max(0, min(S3H, y))
    # For this y, valid x range is [y/S3, 1 - y/S3]
    x_min = y / S3
    x_max = 1 - y / S3
    x = max(x_min, min(x_max, x))
    return np.array([x, y])


def _symmetric_init():
    """Create a symmetric initial configuration for 11 points."""
    # Use a pattern that respects the 3-fold symmetry of the triangle
    # Place points at symmetric positions
    pts = []
    # Center
    pts.append([0.5, S3H / 3])
    # Three points near the midpoints of edges (but inside)
    # Edge midpoints: (0.5, 0), (0, S3H/2*0+...), etc.
    # Actually, let's use a hexagonal-like pattern
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Row 4: 1 point (top)
    # Total: 4+3+2+1+1 = 11
    
    # Bottom row: y=0, x from 0 to 1
    # But we need to be inside, so let's use small y
    y0 = 0.05
    x0_vals = np.linspace(0.1, 0.9, 4)
    for x in x0_vals:
        pts.append([x, y0])
    
    # Row 1: y = S3H/3
    y1 = S3H / 3
    x1_vals = np.linspace(0.15, 0.85, 3)
    for x in x1_vals:
        pts.append([x, y1])
    
    # Row 2: y = 2*S3H/3
    y2 = 2 * S3H / 3
    x2_vals = np.linspace(0.25, 0.75, 2)
    for x in x2_vals:
        pts.append([x, y2])
    
    # Row 3: y = 0.9*S3H
    y3 = 0.9 * S3H
    x3 = 0.5
    pts.append([x3, y3])
    
    # We have 4+3+2+1 = 10, need 1 more
    # Add one more in the center area
    pts.append([0.5, S3H / 2])
    
    return np.array(pts[:N])


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_flat = None
    best_val = -float('inf')
    
    # Try symmetric initialization
    sym_pts = _symmetric_init()
    sym_flat = sym_pts.flatten()
    val = -_penalized_min_area(sym_flat)
    if val > best_val:
        best_val = val
        best_flat = sym_flat.copy()
    
    # Try random restarts with local optimization
    n_restarts = 20
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        # Generate random initial points
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        # Project to ensure inside
        for i in range(N):
            pts[i] = _project_to_triangle(pts[i])
        
        flat = pts.flatten()
        
        # Local optimization with Nelder-Mead
        try:
            result = minimize(
                _penalized_min_area,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < -best_val:
                best_val = -result.fun
                best_flat = result.x.copy()
        except Exception:
            pass
    
    # Final polish on best found
    elapsed = time.time() - start_time
    if best_flat is not None and elapsed < DEADLINE - 10:
        try:
            result = minimize(
                _penalized_min_area,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < -best_val:
                best_val = -result.fun
                best_flat = result.x.copy()
        except Exception:
            pass
    
    points = best_flat.reshape(-1, 2)
    
    # Final projection to ensure all points are inside
    for i in range(len(points)):
        points[i] = _project_to_triangle(points[i])
    
    return points
# EVOLVE-BLOCK-END
