import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    # Vectorized approach for speed
    # There are C(13,3) = 286 triangles
    min_area = float('inf')
    
    # Precompute all indices for vectorization?
    # 286 is small, but let's try to be fast.
    # Actually, a pure Python loop for 286 items is ~0.1ms.
    # 1000 iterations = 100ms.
    # 600s / 0.1ms = 6,000,000 iterations possible.
    # So Python loop is fine.
    
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _get_min_triangle_info(points):
    """
    Returns (min_area, indices_tuple, signed_area) for the smallest triangle.
    """
    min_area = float('inf')
    min_idx = None
    min_signed = 0.0
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Signed area * 2
                signed_area_2 = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
                area = 0.5 * abs(signed_area_2)
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
                    min_signed = signed_area_2
    return min_area, min_idx, min_signed

def _refine_subgradient(points, time_limit=5.0, initial_step=0.05):
    """
    Deterministic subgradient ascent on min triangle area.
    1. Find min triangle (i,j,k).
    2. Compute subgradient of min_area w.r.t. points.
    3. Update points: p_new = p_old + step * grad.
    4. Normalize (scale to unit area hull or fixed size) to prevent drift?
       Actually, the objective is A_min / A_hull.
       Scaling all points by c scales A_min by c^2 and A_hull by c^2, so ratio is invariant.
       We can fix the scale, e.g., A_hull = 1, or just keep it as is and let the ratio guide.
       However, subgradient ascent on A_min/A_hull is tricky because A_hull is non-smooth and global.
       
       Alternative: Optimize A_min subject to A_hull = 1?
       Or just maximize A_min, and rely on the fact that if points expand, A_hull expands too.
       If we maximize A_min, points might fly apart, increasing A_hull faster.
       
       Let's stick to the ratio.
       f(P) = A_min(P) / A_hull(P)
       Gradient of f is complicated.
       
       Simpler heuristic:
       Maximize A_min.
       If points move apart, A_hull increases.
       To keep A_hull constant, we could normalize points after each step?
       If we normalize to A_hull = 1, then we are effectively maximizing A_min under the constraint A_hull=1.
       This is a well-posed problem.
       
       Steps:
       1. Find min triangle.
       2. Compute gradient of A_min w.r.t. points.
       3. Update points.
       4. Compute new A_hull.
       5. Scale points so that A_hull = 1 (or previous A_hull).
       6. Check if A_min increased.
    """
    start_time = time.time()
    current_points = points.copy()
    
    # Initial normalization
    try:
        hull = ConvexHull(current_points)
        current_hull_area = hull.volume
    except Exception:
        return current_points, 0.0
    
    if current_hull_area < 1e-10:
        return current_points, 0.0
        
    # Scale to unit hull area
    scale_factor = np.sqrt(1.0 / current_hull_area)
    current_points = current_points * scale_factor
    current_hull_area = 1.0
    
    current_min_area, current_min_idx, _ = _get_min_triangle_info(current_points)
    current_score = current_min_area / current_hull_area # which is current_min_area
    
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = initial_step
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            break
            
        min_area, min_idx, signed_area = _get_min_triangle_info(current_points)
        
        if min_area < 1e-12:
            # Collinear, add jitter
            current_points[min_idx[0]] += np.random.normal(0, 0.01, 2)
            no_improve_count += 1
            continue
            
        i, j, k = min_idx
        p1 = current_points[i]
        p2 = current_points[j]
        p3 = current_points[k]
        
        # Gradient of Area = 0.5 * | ... |
        # If signed_area > 0, grad w.r.t P is:
        # dA/dp1 = 0.5 * (p2 - p3) rotated 90 deg?
        # Let's derive:
        # A = 0.5 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
        # dA/dx1 = 0.5 * (y2 - y3)
        # dA/dy1 = 0.5 * (x3 - x2)
        # dA/dx2 = 0.5 * (y3 - y1)
        # dA/dy2 = 0.5 * (x1 - x3)
        # dA/dx3 = 0.5 * (y1 - y2)
        # dA/dy3 = 0.5 * (x2 - x1)
        
        # So grad_A = [0.5*(y2-y3), 0.5*(x3-x2)] for p1
        #              [0.5*(y3-y1), 0.5*(x1-x3)] for p2
        #              [0.5*(y1-y2), 0.5*(x2-x1)] for p3
        
        # If signed_area < 0, we are maximizing |A|, so we want to increase A (make it less negative? No, |A| decreases if A->0).
        # Wait. A_min is the *smallest* area. We want to *increase* the smallest area.
        # So we want to move the points such that the area of this specific triangle increases.
        # If signed_area > 0, we want to increase A. Gradient points in direction of increasing A.
        # If signed_area < 0, we want to decrease A (make it more negative, so |A| increases).
        # So we should move in direction -grad_A.
        
        if signed_area > 0:
            g1 = np.array([0.5 * (p2[1] - p3[1]), 0.5 * (p3[0] - p2[0])])
            g2 = np.array([0.5 * (p3[1] - p1[1]), 0.5 * (p1[0] - p3[0])])
            g3 = np.array([0.5 * (p1[1] - p2[1]), 0.5 * (p2[0] - p1[0])])
            sign = 1.0
        else:
            g1 = np.array([0.5 * (p2[1] - p3[1]), 0.5 * (p3[0] - p2[0])])
            g2 = np.array([0.5 * (p3[1] - p1[1]), 0.5 * (p1[0] - p3[0])])
            g3 = np.array([0.5 * (p1[1] - p2[1]), 0.5 * (p2[0] - p1[0])])
            sign = -1.0
            
        new_points = current_points.copy()
        new_points[i] += sign * step_size * g1
        new_points[j] += sign * step_size * g2
        new_points[k] += sign * step_size * g3
        
        # Normalize new points to unit hull area
        try:
            new_hull = ConvexHull(new_points)
            new_hull_area = new_hull.volume
        except Exception:
            no_improve_count += 1
            step_size *= 0.9
            continue
            
        if new_hull_area < 1e-10:
            no_improve_count += 1
            step_size *= 0.9
            continue
            
        scale_new = np.sqrt(1.0 / new_hull_area)
        new_points = new_points * scale_new
        
        new_min_area, _, _ = _get_min_triangle_info(new_points)
        new_score = new_min_area # since hull area is 1
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
            # Increase step size slightly if improving?
            # step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
                
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Uses subgradient ascent with structure-aware initialization.
    """
    start_time = time.time()
    deadline = start_time + 600 # 10 min budget, safety margin inside
    
    best_score = 0.0
    best_points = None
    
    # 1. Generate diverse initial configurations
    configs = []
    
    # Regular polygon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack((np.cos(theta), np.sin(theta))))
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
        
    halton_points = np.zeros((13, 2))
    for i in range(13):
        halton_points[i, 0] = halton(i + 1, 2)
        halton_points[i, 1] = halton(i + 1, 3)
    halton_points = (halton_points - 0.5) * 2.0
    configs.append(halton_points)
    
    # Random uniform
    rng = np.random.default_rng(42)
    configs.append(rng.uniform(-1, 1, (13, 2)))
    
    # Structure-aware: 7 hull, 6 interior
    # Hull points on circle
    hull_theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    hull_pts = np.column_stack((np.cos(hull_theta), np.sin(hull_theta)))
    # Interior points: Halton in square, scaled to fit inside
    int_halton = np.zeros((6, 2))
    for i in range(6):
        int_halton[i, 0] = halton(i + 8, 2)
        int_halton[i, 1] = halton(i + 8, 3)
    int_halton = (int_halton - 0.5) * 1.5 # Scale to fit inside roughly
    mixed = np.vstack((hull_pts, int_halton))
    configs.append(mixed)
    
    # Another structure: 7 hull, 6 interior (random)
    hull_pts2 = np.column_stack((np.cos(hull_theta), np.sin(hull_theta)))
    int_rand = rng.uniform(-0.5, 0.5, (6, 2))
    mixed2 = np.vstack((hull_pts2, int_rand))
    configs.append(mixed2)
    
    # 2. Refine each initial config
    for cfg in configs:
        if time.time() > deadline - 10:
            break
        refined_points, refined_score = _refine_subgradient(cfg, time_limit=10.0, initial_step=0.1)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    # 3. Stochastic search with refinement
    # Use remaining time
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        
        if restart_count % 5 == 0:
            # Random restart
            start_points = rng.uniform(-1, 1, (13, 2))
        else:
            # Perturb best
            noise = rng.normal(0, 0.3, (13, 2))
            start_points = best_points + noise
            
        refined_points, refined_score = _refine_subgradient(start_points, time_limit=5.0, initial_step=0.05)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
