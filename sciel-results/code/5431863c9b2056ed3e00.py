"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Fixed angles for the 14 points. We use a regular 14-gon base.
# This ensures the points are in angular order, which helps maintain convexity 
# if radii are positive and not too disparate.
THETAS = np.linspace(0, 2 * np.pi, N, endpoint=False)
COS_T = np.cos(THETAS)
SIN_T = np.sin(THETAS)

def _points_from_radii(r):
    """Convert radius vector to (N, 2) point array."""
    # r is shape (N,)
    return np.column_stack([r * COS_T, r * SIN_T])

def _hull_area_from_radii(r):
    """
    Compute convex hull area for points defined by radii r at fixed angles THETAS.
    Since points are in angular order and we assume a star-shaped/convex configuration 
    around the origin, the hull is the polygon formed by the points.
    We use the shoelace formula for the polygon area.
    If the polygon is self-intersecting or non-convex, ConvexHull would be safer, 
    but for optimization smoothness, we assume the polygon is convex.
    To be safe, we can compute the area of the polygon P_0...P_{N-1}.
    """
    pts = _points_from_radii(r)
    # Shoelace formula
    x = pts[:, 0]
    y = pts[:, 1]
    # Area = 0.5 * |sum(x_i y_{i+1} - x_{i+1} y_i)|
    # Note: indices wrap around
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross_terms = x * y_next - x_next * y
    area = 0.5 * np.abs(np.sum(cross_terms))
    return area

def _min_triangle_area_from_radii(r):
    """
    Compute minimum triangle area for points defined by radii r.
    """
    pts = _points_from_radii(r)
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _objective_radii(r, beta=200.0):
    """
    Objective function to minimize: HullArea / SoftMinTriangleArea.
    We use a soft-min for the triangle area to make the objective differentiable.
    """
    if np.any(r <= 0):
        return 1e10
    
    hull_area = _hull_area_from_radii(r)
    if hull_area < 1e-10:
        return 1e10
    
    # Compute all triangle areas
    pts = _points_from_radii(r)
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    
    if not areas_list:
        return 1e10
        
    all_areas = np.concatenate(areas_list)
    min_area_hard = np.min(all_areas)
    if min_area_hard < 1e-12:
        return 1e10
    
    # Soft min
    z = -beta * all_areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    return hull_area / soft_min_val

def _objective_hard_radii(r):
    """Hard objective for final evaluation/coordinate descent."""
    if np.any(r <= 0):
        return 1e10
    hull_area = _hull_area_from_radii(r)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area_from_radii(r)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _normalize_unit_area(points):
    """Scale points so convex hull area = 1."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    # Fallback
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def _score_points(points):
    """Compute score for a set of points."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_from_radii(np.linalg.norm(points, axis=1)) # This is only valid if points are on the radial grid
    # Wait, _min_triangle_area_from_radii expects radii. 
    # For general points, we need a general min triangle area function.
    # Let's write a general one to be safe, or just use the one from the seed if we stick to radial.
    # Since we are optimizing radii, the final points are radial.
    # But for safety in the final check, let's use a general vectorized min triangle area.
    
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area / hull_area

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin before 150s kill

    best_r = None
    best_obj = float('inf')
    
    # Initializations for radii
    # 1. Regular polygon (all radii equal)
    r_reg = np.ones(N)
    inits = [r_reg]
    
    # 2. Alternating radii (star-like)
    for alt in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r_alt = np.ones(N)
        r_alt[::2] = 1.0 + alt
        r_alt[1::2] = 1.0 - alt
        inits.append(r_alt)
        
    # 3. Random positive radii
    rng = np.random.default_rng(42)
    for i in range(20):
        r_rand = rng.uniform(0.5, 1.5, N)
        inits.append(r_rand)
        
    # 4. Linear gradient of radii
    r_lin = np.linspace(0.5, 1.5, N)
    inits.append(r_lin)
    
    # Optimization Loop
    for idx, r0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Use L-BFGS-B on the soft objective
        # Bounds: radii must be positive
        bounds = [(1e-3, 10.0)] * N
        
        try:
            res = minimize(
                _objective_radii,
                r0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_obj:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_r = res.x.copy()
        except Exception:
            pass

        # If we have time, do a second pass with higher beta for sharper min
        if time.time() < deadline - 10:
            try:
                # We can't easily change beta inside minimize without a wrapper, 
                # but we can just run another optimization from the current best if it improved
                if best_r is not None:
                    res2 = minimize(
                        _objective_radii,
                        best_r,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res2.fun < best_obj:
                        best_obj = res2.fun
                        best_r = res2.x.copy()
            except Exception:
                pass

    # Final Refinement: Coordinate Descent on Radii
    if best_r is not None and time.time() < deadline - 10:
        r_cd = best_r.copy()
        step = 0.01
        improved = True
        rounds = 0
        while improved and rounds < 50 and time.time() < deadline - 5:
            improved = False
            rounds += 1
            for i in range(N):
                for _ in range(10): # Binary search for best step
                    r_plus = r_cd.copy()
                    r_plus[i] += step
                    if r_plus[i] > 0:
                        obj_plus = _objective_hard_radii(r_plus)
                    else:
                        obj_plus = 1e10
                    
                    r_minus = r_cd.copy()
                    r_minus[i] -= step
                    if r_minus[i] > 0:
                        obj_minus = _objective_hard_radii(r_minus)
                    else:
                        obj_minus = 1e10
                        
                    curr_obj = _objective_hard_radii(r_cd)
                    
                    if obj_plus < curr_obj:
                        r_cd = r_plus
                        improved = True
                    elif obj_minus < curr_obj:
                        r_cd = r_minus
                        improved = True
                    else:
                        break
                    step *= 0.5
            step = 0.01 # Reset step size for next coordinate
            
        if _objective_hard_radii(r_cd) < best_obj:
            best_obj = _objective_hard_radii(r_cd)
            best_r = r_cd

    # Fallback if optimization failed
    if best_r is None:
        best_r = np.ones(N)

    # Convert best radii to points
    final_points = _points_from_radii(best_r)
    
    # Normalize to unit hull area
    final_points = _normalize_unit_area(final_points)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            # Fallback to regular polygon
            final_points = _points_from_radii(np.ones(N))
            final_points = _normalize_unit_area(final_points)
    except Exception:
        final_points = _points_from_radii(np.ones(N))
        final_points = _normalize_unit_area(final_points)

    return final_points

# EVOLVE-BLOCK-END
