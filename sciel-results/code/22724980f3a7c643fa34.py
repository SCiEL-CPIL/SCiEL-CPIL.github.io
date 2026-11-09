# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """
    Compute the minimum vertex score for a given set of 20 points.
    Assumes points are in cyclic order and form a convex polygon.
    """
    n = len(points)
    
    # 1. Normalization
    centroid = points.mean(axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_d = np.mean(dists_to_centroid)
    
    if mean_d <= 0.01:
        return 0.0
        
    centered = centered / mean_d
    
    # 2. Quick Convexity & Validity Check
    # Check for repeated points
    min_dist = np.min(np.linalg.norm(centered[:, np.newaxis, :] - centered[np.newaxis, :, :], axis=2).diagonal(offset=1))
    if min_dist < 1e-10:
        return 0.0
        
    # Check cross products for consistent turning
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_products[i] = e1[0]*e2[1] - e1[1]*e2[0]
        
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
        
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
        
    # Check total turning angle is approx 2*pi
    # For a strictly convex polygon, the sum of exterior angles is 2*pi.
    # We can verify this by summing signed angles.
    total_turn = 0.0
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i+1) % n]
        p2 = centered[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos_a = np.dot(e1, e2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        a = np.arccos(cos_a)
        if cross_products[i] < 0:
            a = -a
        total_turn += a
        
    if abs(abs(total_turn) - 2*np.pi) > 1e-5:
        return 0.0

    # 3. Compute Vertex Scores
    min_score = 1.0
    
    # Precompute all pairwise distances
    # dist_matrix[i, j] is distance between point i and point j
    dist_matrix = np.linalg.norm(centered[:, np.newaxis, :] - centered[np.newaxis, :, :], axis=2)
    
    for i in range(n):
        # Distances to other vertices
        d = dist_matrix[i].copy()
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        # Contract: d_0 <= ... <= d_18
        # If any is < 0.05, score 0
        if d_sorted[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
            
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        if vertex_score < min_score:
            min_score = vertex_score
            
    return min_score

def _score_from_base_angles(base_angles):
    """
    Construct the 20-gon from 5 base angles with 4-fold symmetry and compute score.
    base_angles: array of 5 angles in [0, pi/2)
    """
    n_base = 5
    # Generate all 20 angles
    # angles = [b_j + k*pi/2 for j in 0..4 for k in 0..3]
    offsets = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    all_angles = np.concatenate([base_angles[:, np.newaxis] + offsets[np.newaxis, :]])
    all_angles = all_angles.flatten()
    
    # Sort to ensure cyclic order
    all_angles = np.sort(all_angles)
    
    # Construct points on unit circle
    x = np.cos(all_angles)
    y = np.sin(all_angles)
    points = np.column_stack((x, y))
    
    return _compute_score(points)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0 # Safety margin
    
    n_base = 5
    bounds = [(0.0, np.pi/2)] * n_base
    
    best_score = 0.0
    best_base_angles = np.linspace(0, np.pi/2, n_base, endpoint=False)
    
    # 1. Global Optimization with Differential Evolution
    # We maximize score, so we minimize negative score
    def objective(base_angles):
        if time.time() > deadline:
            return 0.0
        s = _score_from_base_angles(base_angles)
        return -s
        
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < 0: # Valid improvement
            cand_score = -result.fun
            if cand_score > best_score:
                best_score = cand_score
                best_base_angles = result.x.copy()
    except Exception as e:
        pass

    # 2. Local Refinement
    # Try local optimization from the best global solution and several random starts
    local_starts = [best_base_angles]
    rng = np.random.default_rng(123)
    for _ in range(10):
        local_starts.append(rng.uniform(0, np.pi/2, n_base))
    
    def local_objective(base_angles):
        if time.time() > deadline:
            return 0.0
        return -_score_from_base_angles(base_angles)
        
    for start in local_starts:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust for small dimensions
            res = minimize(
                local_objective,
                start,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.fun < 0:
                cand_score = -res.fun
                if cand_score > best_score:
                    best_score = cand_score
                    best_base_angles = res.x.copy()
        except Exception:
            pass
            
        # Also try L-BFGS-B if we have time
        if time.time() < deadline - 20:
            try:
                res2 = minimize(
                    local_objective,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res2.fun < 0:
                    cand_score = -res2.fun
                    if cand_score > best_score:
                        best_score = cand_score
                        best_base_angles = res2.x.copy()
            except Exception:
                pass

    # Construct final polygon
    offsets = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    all_angles = np.concatenate([best_base_angles[:, np.newaxis] + offsets[np.newaxis, :]])
    all_angles = all_angles.flatten()
    all_angles = np.sort(all_angles)
    
    x = np.cos(all_angles)
    y = np.sin(all_angles)
    points = np.column_stack((x, y))
    
    # Final normalization check (though _compute_score does it internally, 
    # the task requires the returned array to be normalized? 
    # "The polygon is normalized: subtract the centroid, divide by the mean distance to it...")
    # The task says "returning a (20, 2) array of finite vertex coordinates in cyclic order."
    # And "The polygon is normalized: ... It must then be strictly convex..."
    # Usually, the evaluator normalizes the input to check validity/score.
    # However, if I return the raw unit-circle points, the evaluator will normalize them.
    # Let's return the points as constructed on the unit circle, which will be normalized by the evaluator.
    # Or should I return the normalized ones?
    # "Write equidistant_points_polygon20() ... returning a (20, 2) array ... The polygon is normalized: subtract..."
    # This phrasing suggests the evaluator performs the normalization on the returned array.
    # So returning the unit circle points is safe.
    
    return points

# EVOLVE-BLOCK-END
