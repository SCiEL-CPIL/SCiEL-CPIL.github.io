"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _time_left(deadline):
    return deadline - time.time()

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _all_triangle_areas(points):
    """
    Compute areas of all triangles.
    """
    areas = []
    x = points[:, 0]
    y = points[:, 1]
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas.append(0.5 * np.abs(cross))
    if not areas:
        return np.array([])
    return np.concatenate(areas)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(x):
    """
    Objective: Minimize -min_triangle_area.
    We assume x is normalized to unit hull area externally or we penalize deviation.
    Actually, for SLSQP, it's better to normalize inside the objective/constraints 
    or fix the scale. 
    Let's normalize x to unit area inside the evaluation for the objective, 
    but SLSQP works on raw coordinates. 
    To make it well-conditioned, we will scale x such that hull area is 1.
    However, SLSQP steps might break the hull. 
    Alternative: Keep points in a bounded region, e.g., inside unit circle, 
    and normalize at the end. But the score depends on the ratio.
    
    Let's use a penalty approach or just maximize min_area / hull_area directly?
    SLSQP handles constraints well.
    
    Strategy:
    1. Normalize points to unit area.
    2. Objective: -min_triangle_area(normalized_points)
    3. Constraints: 
       - Hull area = 1 (This is tricky as a constraint if we normalize inside).
       
    Better Strategy for SLSQP:
    Work in raw coordinates.
    Objective: - (min_triangle_area / hull_area)
    This is non-smooth due to min and division.
    
    Let's stick to the "Target Threshold" approach which is a structural change.
    We want to find max t such that there exists a configuration with min_area >= t * hull_area.
    
    Let's define the optimization in terms of raw coordinates p.
    Let A_min(p) be min triangle area.
    Let A_hull(p) be hull area.
    We want to maximize A_min(p) / A_hull(p).
    
    Since scaling p by s scales A_min by s^2 and A_hull by s^2, the ratio is scale-invariant.
    We can fix the scale, e.g., sum of squared distances = 1, or hull area = 1.
    
    Let's use constraints:
    1. Hull Area = 1. (Equality constraint)
    2. All triangle areas >= t. (Inequality constraints)
    
    If we fix Hull Area = 1, then maximizing min triangle area is equivalent to maximizing the score.
    However, maintaining Hull Area = 1 as an equality constraint in SLSQP can be stiff.
    
    Alternative:
    Objective: -min_triangle_area
    Constraints:
    1. Hull Area <= 1 (We want to maximize min_area, so we want to keep hull small? No, we want to maximize min_area/hull_area. If we fix hull_area=1, we just maximize min_area. If we allow hull_area < 1, we might shrink the whole thing, but min_area shrinks too. The ratio is invariant. So we can fix hull_area = 1.)
    
    Let's try:
    Objective: -min_triangle_area(x)
    Constraints:
    - Hull_Area(x) == 1
    
    This is a hard constraint. SLSQP can handle it.
    
    To handle the non-smoothness of min_triangle_area, we can use the log-sum-exp trick or just rely on SLSQP's robustness with a good initial guess. The min function is piecewise linear (in terms of cross products), so gradients exist almost everywhere.
    
    We will implement _objective and _constraints for SLSQP.
    """
    points = x.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    
    # Calculate areas
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 1e10
        
    min_a = np.min(areas)
    
    # We want to minimize -min_a
    return -min_a

def _constraints(x):
    points = x.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    
    # Constraint: Hull Area == 1
    try:
        hull_area = _hull_area(points)
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    # We want hull_area - 1 == 0
    return hull_area - 1.0

def _generate_initial_points():
    """
    Generate multiple diverse initial points.
    1. Regular polygon
    2. Jittered grid
    3. Random
    """
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    h = ConvexHull(p1)
    p1 = p1 * np.sqrt(1.0 / h.volume)
    inits.append(p1.flatten())
    
    # Jittered Grid
    np.random.seed(42)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])[:N]
    pts += np.random.uniform(-0.1, 0.1, pts.shape)
    h = ConvexHull(pts)
    pts = pts * np.sqrt(1.0 / h.volume)
    inits.append(pts.flatten())
    
    # Random
    np.random.seed(43)
    pts = np.random.uniform(-1, 1, (N, 2))
    h = ConvexHull(pts)
    pts = pts * np.sqrt(1.0 / h.volume)
    inits.append(pts.flatten())
    
    # Concentric rings
    n1, n2 = 7, 7
    r1, r2 = 1.0, 0.5
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * pi, n2, endpoint=False) + np.pi/n2
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    pts = np.vstack([p1, p2])
    h = ConvexHull(pts)
    pts = pts * np.sqrt(1.0 / h.volume)
    inits.append(pts.flatten())
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score_val = -1e10
    
    inits = _generate_initial_points()
    
    for x0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # SLSQP with Hull Area = 1 constraint
            # We want to maximize min_triangle_area.
            # Objective: -min_triangle_area
            # Constraint: hull_area - 1 = 0
            
            # To help SLSQP, we might need to relax the equality constraint slightly 
            # or use a penalty if it struggles. 
            # Let's try strict equality first.
            
            # Bounds: Keep points within a reasonable box to prevent explosion
            # If hull area is 1, points shouldn't go too far.
            # Regular polygon of 14 points has radius ~1.4. 
            # Let's allow up to 10.
            bounds = [(-10, 10)] * (2 * N)
            
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': _constraints}],
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or _score(res.x.reshape(N, 2)) > best_score_val:
                curr_score = _score(res.x.reshape(N, 2))
                if curr_score > best_score_val:
                    best_score_val = curr_score
                    best_result = res.x.copy()
                    
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        h = ConvexHull(best_result.reshape(N, 2))
        best_result = best_result * np.sqrt(1.0 / h.volume)
        
    final_points = best_result.reshape(N, 2)
    
    # Final Normalization to ensure unit hull area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        
    return final_points
# EVOLVE-BLOCK-END
