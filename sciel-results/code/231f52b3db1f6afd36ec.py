import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _fast_score(points):
    """
    Vectorized computation of min triangle area and convex hull area.
    Returns (min_area, hull_area, min_indices_list).
    """
    n = len(points)
    if n != 13:
        return 0.0, 1.0, []
    
    if not np.all(np.isfinite(points)):
        return 0.0, 1.0, []

    # Vectorized triangle area calculation
    # We need all 286 triangles.
    # Indices:
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Flatten and filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * | (p2 - p1) x (p3 - p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    # Find all triangles that are "close" to the minimum (active set)
    # Tolerance: 1% of min_area or absolute small value
    tol = max(1e-8, 0.01 * min_area)
    active_mask = areas <= (min_area + tol)
    active_indices = list(zip(i_flat[active_mask], j_flat[active_mask], k_flat[active_mask]))
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0, 1.0, []
        
    if hull_area <= 1e-10:
        return 0.0, 1.0, []

    return min_area, hull_area, active_indices

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    min_area, hull_area, _ = _fast_score(points)
    if min_area == 0.0:
        return 0.0
    return min_area / hull_area

def _generate_initial_points():
    """
    Generate initial configurations based on the known optimal structure:
    A ring of points and a cluster in the center.
    """
    configs = []
    
    # 1. 7 outer points (heptagon) + 6 inner points (hexagon)
    # This is a common optimal topology for N=13
    theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(theta_outer), np.sin(theta_outer)))
    
    # Inner points scaled and rotated
    r_inner = 0.5
    theta_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack((r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)))
    
    configs.append(np.vstack([outer, inner]))
    
    # 2. 6 outer + 7 inner
    theta_outer2 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer2 = np.column_stack((np.cos(theta_outer2), np.sin(theta_outer2)))
    r_inner2 = 0.6
    theta_inner2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner2 = np.column_stack((r_inner2 * np.cos(theta_inner2), r_inner2 * np.sin(theta_inner2)))
    configs.append(np.vstack([outer2, inner2]))
    
    # 3. Random restarts
    rng = np.random.default_rng(123)
    for _ in range(5):
        # Random points, but ensure some spread
        pts = rng.uniform(-1, 1, (13, 2))
        # Normalize to unit circle roughly
        norms = np.linalg.norm(pts, axis=1)
        norms[norms < 1e-6] = 1
        pts = pts / norms[:, np.newaxis] * rng.uniform(0.5, 1.0, (13, 1))
        configs.append(pts)
        
    # 4. Halton sequence
    def halton(n, base):
        result = []
        f = 1
        for i in range(1, n + 1):
            r = 0
            f = 1
            index = i
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            result.append(r)
        return np.array(result)
    
    x = halton(13, 2)
    y = halton(13, 3)
    pts = np.column_stack((x, y))
    pts = (pts - 0.5) * 2.0
    configs.append(pts)
    
    return configs

def _refine_max_min(points, time_limit=10.0):
    """
    Refine using subdifferential ascent on the active set of minimum triangles.
    """
    start_time = time.time()
    current_points = points.copy()
    current_min, current_hull, _ = _fast_score(current_points)
    current_score = current_min / current_hull if current_hull > 0 else 0
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < time_limit:
        min_area, hull_area, active_indices = _fast_score(current_points)
        
        if min_area < 1e-8:
            # Collinear or nearly so
            current_points += np.random.normal(0, 0.01, current_points.shape)
            new_score = _score(current_points)
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            continue
        
        # Compute subgradient for each point
        # The objective is min(areas). The subdifferential is the convex hull of gradients of active areas.
        # We approximate by the mean gradient of the active set.
        
        grads = np.zeros_like(current_points)
        
        if not active_indices:
            break
            
        for (i, j, k) in active_indices:
            p1 = current_points[i]
            p2 = current_points[j]
            p3 = current_points[k]
            
            # Cross product sign
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            sign = np.sign(cross)
            if sign == 0:
                sign = 1.0 # Avoid zero gradient direction issues, randomize if needed later
            
            # Gradients of area w.r.t p1, p2, p3
            # Area = 0.5 * sign * ( (p2-p1) x (p3-p1) )
            # dA/dp1 = -0.5 * sign * rot90(p3 - p2)
            # dA/dp2 =  0.5 * sign * rot90(p3 - p1)
            # dA/dp3 =  0.5 * sign * rot90(p1 - p2)
            
            def rot90(v):
                return np.array([-v[1], v[0]])
            
            g1 = -0.5 * sign * rot90(p3 - p2)
            g2 =  0.5 * sign * rot90(p3 - p1)
            g3 =  0.5 * sign * rot90(p1 - p2)
            
            grads[i] += g1
            grads[j] += g2
            grads[k] += g3
            
        # Average the gradients for points involved in multiple active triangles
        # We need to know which points are active
        active_points = set()
        for (i, j, k) in active_indices:
            active_points.add(i)
            active_points.add(j)
            active_points.add(k)
            
        for idx in active_points:
            # Normalize the gradient direction for this point? 
            # Actually, summing them gives the subdifferential direction.
            # We just use the sum.
            pass
            
        # Update points
        new_points = current_points.copy()
        for idx in active_points:
            new_points[idx] += step_size * grads[idx]
            
        new_min, new_hull, _ = _fast_score(new_points)
        new_score = new_min / new_hull if new_hull > 0 else 0
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.98
        else:
            no_improve_count += 1
            step_size *= 0.8
            if step_size < 1e-6:
                break
            if no_improve_count > max_no_improve:
                break
                
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Uses a hybrid approach:
    1. Generate diverse initial configurations.
    2. Apply max-min subdifferential refinement.
    3. Stochastic restarts with refinement.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Initial configurations
    configs = _generate_initial_points()
    
    # 2. Refine each initial config
    for cfg in configs:
        if time.time() > deadline:
            break
        refined_points, refined_score = _refine_max_min(cfg, time_limit=5.0)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    # 3. Stochastic search with refinement
    rng = np.random.default_rng(42)
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        
        if restart_count % 5 == 0:
            # Random restart
            start_points = rng.uniform(-1, 1, (13, 2))
            # Normalize
            norms = np.linalg.norm(start_points, axis=1)
            norms[norms < 1e-6] = 1
            start_points = start_points / norms[:, np.newaxis] * rng.uniform(0.5, 1.0, (13, 1))
        else:
            # Perturb best
            noise = rng.normal(0, 0.5, (13, 2))
            start_points = best_points + noise
            
        refined_points, refined_score = _refine_max_min(start_points, time_limit=3.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
