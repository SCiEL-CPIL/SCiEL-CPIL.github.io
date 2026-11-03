import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRI = len(IDX_I)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _refine_remaining(points, anchor_indices, anchor_points, time_limit=5.0):
    """
    Fix anchor points, optimize the rest to maximize the minimum area 
    among all triangles that do NOT consist entirely of anchor points.
    This is a proxy for maximizing the second minimum area.
    """
    start_time = time.time()
    
    # Identify triangles to consider
    # A triangle is "active" if it involves at least one non-anchor point.
    # If it involves 2 or 3 non-anchor points, or 1 non-anchor and 2 anchors, it's active.
    # The only inactive triangle is the one formed by the 3 anchors themselves.
    # We want to maximize the min area over active triangles.
    
    # For simplicity and speed, we maximize the min area over ALL triangles 
    # except the anchor triangle. But since the anchor triangle area is fixed 
    # (by fixing anchor points), maximizing min over all triangles 
    # is equivalent to maximizing min over active triangles 
    # IF the anchor triangle area is not the global minimum.
    # Actually, we want to ensure the second minimum is as large as possible.
    # So we maximize min(areas[active]).
    
    active_mask = np.ones(N_TRI, dtype=bool)
    # Find the index of the triangle formed by the 3 anchors
    # Sort anchors to match IDX_I, IDX_J, IDX_K order
    sorted_anchors = np.sort(anchor_indices)
    for i in range(N_TRI):
        if IDX_I[i] == sorted_anchors[0] and IDX_J[i] == sorted_anchors[1] and IDX_K[i] == sorted_anchors[2]:
            active_mask[i] = False
            break
            
    # If no active triangles (impossible for N=14), return
    if not np.any(active_mask):
        return points
        
    non_anchor_idx = [i for i in range(N) if i not in anchor_indices]
    n_non_anchor = len(non_anchor_idx)
    
    if n_non_anchor == 0:
        return points
        
    # Objective: minimize -min(areas[active_mask])
    # But min is non-differentiable. Use soft min or just hard min with Nelder-Mead?
    # Nelder-Mead on hard min is robust for small dimensions.
    # 11 points * 2 coords = 22 dimensions.
    
    def obj(flat):
        if time.time() - start_time > time_limit:
            return 1e10 # Time out penalty
        pts = points.copy()
        pts[non_anchor_idx] = flat.reshape(n_non_anchor, 2)
        areas = _all_areas(pts)
        min_active = np.min(areas[active_mask])
        return -min_active

    # Initial guess for non-anchor points
    x0 = points[non_anchor_idx].flatten()
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.success or res.fun < 1e9:
            new_pts = points.copy()
            new_pts[non_anchor_idx] = res.x.reshape(n_non_anchor, 2)
            return new_pts
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _elapsed():
        return time.time() - start_time

    # --- Phase 1: Generate diverse starting points ---
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    starts = [np.column_stack((np.cos(angles), np.sin(angles)))]
    
    # 2. Random points in unit circle
    np.random.seed(42)
    for _ in range(20):
        r = np.sqrt(np.random.uniform(0, 1, N))
        th = np.random.uniform(0, 2 * np.pi, N)
        pts = np.column_stack((r * np.cos(th), r * np.sin(th)))
        starts.append(pts)
        
    # 3. Ring + Center
    for r_ring in [0.5, 0.8, 1.0]:
        angles_ring = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts_ring = np.column_stack((r_ring * np.cos(angles_ring), r_ring * np.sin(angles_ring)))
        pts_center = np.zeros((2, 2))
        # Jitter center slightly
        pts_center[0] += np.random.uniform(-0.1, 0.1, 2)
        pts_center[1] += np.random.uniform(-0.1, 0.1, 2)
        starts.append(np.vstack([pts_ring, pts_center]))

    # --- Phase 2: Iterative Anchor Repair ---
    for start in starts:
        if _elapsed() > deadline - 20.0:
            break
            
        current_pts = start.copy()
        current_score = _score(current_pts)
        update_best(current_pts, current_score)
        
        # Iterate repair steps
        for step in range(10):
            if _elapsed() > deadline - 10.0:
                break
                
            areas = _all_areas(current_pts)
            min_idx = np.argmin(areas)
            
            # Identify the 3 vertices of the worst triangle
            i, j, k = IDX_I[min_idx], IDX_J[min_idx], IDX_K[min_idx]
            anchor_indices = [i, j, k]
            anchor_points = current_pts[anchor_indices].copy()
            
            # 1. Push anchors apart to increase their triangle area
            # Gradient of area w.r.t. points:
            # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
            # Let's just move them slightly outward from the centroid of the triangle
            # or use a simple gradient step.
            # Since we are fixing them for the next step, we want to maximize their area.
            # But we also want to keep them in a good position for the rest.
            # A simple heuristic: move each point slightly away from the centroid of the triangle.
            centroid = (current_pts[i] + current_pts[j] + current_pts[k]) / 3.0
            for idx in anchor_indices:
                diff = current_pts[idx] - centroid
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    # Push out by a small amount
                    current_pts[idx] += 0.05 * (diff / norm)
            
            # 2. Refine the other 11 points, keeping the 3 anchors fixed
            # We pass the current positions of anchors, but we want to optimize the rest.
            # _refine_remaining will fix the anchors at their CURRENT positions (after push).
            refined_pts = _refine_remaining(current_pts, anchor_indices, current_pts[anchor_indices], time_limit=5.0)
            
            new_score = _score(refined_pts)
            if new_score > current_score:
                current_pts = refined_pts
                current_score = new_score
                update_best(current_pts, current_score)
            else:
                # If it didn't improve, maybe the push was bad. Break or try a different anchor?
                # For simplicity, break and try next start.
                break
                
    # --- Phase 3: Final Polish with Full Optimization ---
    if best_points is not None and _elapsed() < deadline - 5.0:
        # Do a final Nelder-Mead on the full 28D space to polish the best found
        def obj_full(flat):
            pts = flat.reshape(N, 2)
            return -_score(pts)
            
        x0 = best_points.flatten()
        try:
            res = minimize(obj_full, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.success:
                final_pts = res.x.reshape(N, 2)
                final_score = _score(final_pts)
                update_best(final_pts, final_score)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
