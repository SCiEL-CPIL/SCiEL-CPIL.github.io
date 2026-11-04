import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14

# Precompute triple indices
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(points):
    """
    Objective function for minimization.
    Returns negative score.
    """
    s = _score(points)
    return -s

def _generate_ring_points(r, theta_offset):
    """
    Generate 14 points: 7 on unit circle, 7 on circle of radius r.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = theta_offset + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0
    best_points = None
    candidates = []

    # 1. Regular 14-gon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    candidates.append(init_reg)

    # 2. Grid search for 7+7 rings to find a good starting point
    # This provides a strong symmetric candidate
    best_ring_score = 0
    best_ring_pts = None
    r_vals = np.linspace(0.1, 0.9, 15)
    theta_vals = np.linspace(0, np.pi/7, 10, endpoint=False)
    
    for r in r_vals:
        for theta in theta_vals:
            if time.time() > deadline - 30:
                break
            pts = _generate_ring_points(r, theta)
            s = _score(pts)
            if s > best_ring_score:
                best_ring_score = s
                best_ring_pts = pts.copy()
        
        if time.time() > deadline - 30:
            break
            
    if best_ring_pts is not None:
        candidates.append(best_ring_pts)
    
    # 3. Add a few random candidates to break symmetry
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        pts = np.random.rand(N, 2) * 2 - 1
        # Normalize to unit circle roughly
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        candidates.append(pts)

    # 4. Local optimization on the best candidates
    # We will pick the top 3 unique candidates based on initial score
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate roughly (optional, but good for time)
    unique_candidates = []
    for s, pts in scored_candidates:
        if not unique_candidates:
            unique_candidates.append((s, pts))
        else:
            # Check if too similar to existing best
            too_similar = False
            for _, existing_pts in unique_candidates:
                dist = np.linalg.norm(pts - existing_pts)
                if dist < 0.1:
                    too_similar = True
                    break
            if not too_similar:
                unique_candidates.append((s, pts))
        
        if len(unique_candidates) >= 3:
            break

    # Optimize each unique candidate
    for s_init, pts_init in unique_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use Powell method, which is derivative-free and robust
            # maxiter is limited to ensure we finish in time
            res = minimize(_neg_score, pts_init, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
            
            if res.success or res.fun < -s_init:
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = res.x.copy()
        except Exception:
            pass

    # If no optimization succeeded or improved, use the best initial
    if best_points is None:
        if scored_candidates:
            best_score = scored_candidates[0][0]
            best_points = scored_candidates[0][1]
        else:
            best_points = init_reg
            best_score = _score(init_reg)
            
    return best_points
