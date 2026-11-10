import numpy as np
import time
from scipy.optimize import minimize

_N = 20
_BASE_ANGLES = np.linspace(0, 2 * np.pi, _N, endpoint=False)

def _compute_scores(points):
    """
    Calculate vertex scores for a given set of points (N, 2).
    Returns array of N scores.
    """
    n = len(points)
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity so it's not considered in sorting/mins
    dists_self_inf = dists.copy()
    np.fill_diagonal(dists_self_inf, np.inf)
    
    # Sort distances for each vertex
    # sorted_d[i] contains the 20 distances from i to all other vertices, sorted
    # The first element is 0 (distance to self, but we set it to inf, so it goes to end? 
    # Wait, np.fill_diagonal with inf means self-distance is inf.
    # So sorted_d[i, 0] is the min distance to another vertex.
    # We need distances to the *other* 19 vertices.
    # Since self is inf, it will be at index 19.
    # So d_0 ... d_18 are the distances to the other 19 vertices.
    sorted_d = np.sort(dists_self_inf, axis=1)
    d = sorted_d[:, :n-1] # Shape (N, 19)
    
    scores = np.zeros(n)
    
    # Check for repeated points or very close points
    # d_sorted[0] is the closest other vertex
    has_close = d[:, 0] < 0.05
    
    # Calculate gaps
    # d shape (N, 19). Gaps shape (N, 18)
    gaps = d[:, 1:] - d[:, :-1]
    
    # Find k: index of min gap
    # If multiple, argmin picks the first one
    k_idx = np.argmin(gaps, axis=1)
    
    # Calculate D
    # d[np.arange(n), k_idx] gives d_k
    # d[np.arange(n), k_idx + 1] gives d_{k+1}
    D = (d[np.arange(n), k_idx] + d[np.arange(n), k_idx + 1]) / 2.0
    
    valid = (~has_close) & (D >= 0.1)
    
    selected = np.zeros((n, 4))
    
    # Masks for different cases of k
    # k ranges from 0 to 17 (since there are 18 gaps between 19 elements)
    # Case 1: k == 0
    mask_k0 = valid & (k_idx == 0)
    # Case 2: k == 17 (max index for gaps)
    mask_k17 = valid & (k_idx == 17)
    # Case 3: 0 < k < 17
    mask_mid = valid & (k_idx > 0) & (k_idx < 17)
    
    # Fill selected distances
    # For k=0: d_0..d_3
    if np.any(mask_k0):
        selected[mask_k0] = d[mask_k0][:, 0:4]
        
    # For k=17: d_15..d_18
    if np.any(mask_k17):
        selected[mask_k17] = d[mask_k17][:, 15:19]
        
    # For mid: d_{k-1}..d_{k+2}
    if np.any(mask_mid):
        idxs = np.where(mask_mid)[0]
        for i in idxs:
            k = k_idx[i]
            selected[i] = d[i, k-1:k+3]
            
    # Calculate raw score
    # ratios = max(d/D, D/d)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.maximum(selected / D[:, np.newaxis], D[:, np.newaxis] / selected)
    
    raw = np.mean(ratios, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        score_vals = np.minimum(1.0, 1.0 / raw)
        
    scores = np.where(valid, score_vals, 0.0)
    
    return scores

def _check_convexity(points):
    """
    Check if points form a strictly convex polygon in cyclic order.
    Returns True if valid, False otherwise.
    """
    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    
    e1 = p_next - p
    e2 = p_next2 - p_next
    
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Strictly convex: no zero cross, all same sign
    if np.any(np.abs(cross) < 1e-9):
        return False
    if np.any(cross > 0) and np.any(cross < 0):
        return False
    return True

def _min_vertex_score(params):
    """
    params: array of length 2*N (radii + angle_deltas).
    Returns min vertex score after normalization. 0 if invalid.
    """
    n = _N
    radii = params[:n]
    angle_deltas = params[n:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 0.01)
    
    angles = _BASE_ANGLES + angle_deltas
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points = points / mean_dist
    
    # Check convexity
    if not _check_convexity(points):
        return 0.0
        
    scores = _compute_scores(points)
    return float(np.min(scores))

def _score_and_penalty(params):
    """
    Returns (score, penalty).
    Score is min vertex score.
    Penalty is sum of squared negative cross products (convexity violation).
    """
    n = _N
    radii = params[:n]
    angle_deltas = params[n:]
    
    radii = np.maximum(radii, 0.01)
    
    angles = _BASE_ANGLES + angle_deltas
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0, 100.0
        
    points = points / mean_dist
    
    # Calculate convexity penalty
    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    e1 = p_next - p
    e2 = p_next2 - p_next
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # If any cross is negative, it's a violation (assuming CCW order)
    # We want all crosses to be positive (or all negative).
    # Let's assume we want positive. If the initial regular polygon gives positive, we stick to that.
    # Regular polygon CCW: cross > 0.
    pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
    
    scores = _compute_scores(points)
    score = float(np.min(scores))
    
    return score, pen

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0 # 20s safety margin
    
    def objective(params):
        score, pen = _score_and_penalty(params)
        # We want to maximize score, so minimize -score.
        # Add penalty to avoid non-convex shapes.
        return -score + 50.0 * pen
        
    best_score = -1.0
    best_params = None
    
    # Initializations
    # 1. Regular polygon (radii=1, angles=0)
    init1 = np.concatenate([np.ones(_N), np.zeros(_N)])
    
    # 2. Radial perturbations
    rng = np.random.default_rng(42)
    
    inits = [init1]
    
    # Add some radial-only perturbations (angle deltas = 0)
    for _ in range(10):
        r_pert = 1.0 + 0.1 * rng.standard_normal(_N)
        r_pert = np.maximum(r_pert, 0.5)
        a_pert = np.zeros(_N)
        inits.append(np.concatenate([r_pert, a_pert]))
        
    # Add some joint perturbations
    for _ in range(10):
        r_pert = 1.0 + 0.1 * rng.standard_normal(_N)
        r_pert = np.maximum(r_pert, 0.5)
        a_pert = 0.05 * rng.standard_normal(_N)
        # Center angle perturbations to maintain overall rotation roughly
        a_pert -= np.mean(a_pert)
        inits.append(np.concatenate([r_pert, a_pert]))
        
    # Add Fourier-like perturbations
    for freq in [2, 4, 5]:
        r_pert = 1.0 + 0.1 * np.cos(freq * _BASE_ANGLES)
        a_pert = 0.05 * np.sin(freq * _BASE_ANGLES)
        inits.append(np.concatenate([r_pert, a_pert]))
        
    for x0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            # Evaluate the actual score of the result
            sc = _min_vertex_score(res.x)
            
            if sc > best_score:
                best_score = sc
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to regular
        best_params = init1
        best_score = _min_vertex_score(best_params)
        
    # Construct final polygon from best_params
    radii = best_params[:_N]
    angle_deltas = best_params[_N:]
    
    radii = np.maximum(radii, 0.01)
    angles = _BASE_ANGLES + angle_deltas
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.column_stack((x, y))
    
    # Final Normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
        
    return polygon
