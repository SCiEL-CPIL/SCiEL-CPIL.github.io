# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    if d[0] < 0.05:
        return 0.0
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _objective(params):
    """
    Objective function to minimize. 
    We want to maximize the minimum vertex score.
    So we minimize -min_score + penalty.
    params: 20 angles.
    """
    n = 20
    angles = params
    # Ensure angles are sorted and cover 2*pi? 
    # Actually, for a convex polygon containing the origin, angles should be monotonically increasing.
    # Let's just use the angles as is, assuming they are sorted in the search space.
    # To ensure convexity and "star-shaped" w.r.t origin, we assume angles are sorted.
    # We will sort them in the evaluation to be safe, but the optimizer works better if the space is continuous.
    # Let's define the polygon by angles theta_i and radius 1 (normalized later).
    # Wait, the normalization subtracts centroid and divides by mean distance.
    # If we fix radius to 1 for all vertices, it's a regular polygon.
    # We need to vary the angles to change the shape.
    # If we vary angles, the polygon is no longer "equidistant from origin" but the normalization handles the scale.
    # However, if we just vary angles, the polygon might become non-convex if angles are not sorted or if we wrap around badly.
    
    # Let's assume params are 20 angles in [0, 2*pi).
    # We sort them to ensure cyclic order.
    # But sorting makes the function non-smooth.
    # Alternative: params are deviations from regular angles, constrained to keep order.
    # Let's try: theta_i = i * (2*pi/20) + params[i].
    # We must ensure theta_i < theta_{i+1}.
    # This is a constraint.
    
    # Simpler approach for Nelder-Mead: 
    # Use 20 angles, sort them. If the spread is not close to 2*pi, penalize.
    # Actually, the most robust way for Nelder-Mead with sorting is to penalize non-convexity heavily.
    
    angles = np.sort(params)
    # Ensure the polygon is closed and covers the circle.
    # Add 2*pi to the last angle to compute edge vectors?
    # No, just compute points.
    x = np.cos(angles)
    y = np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    # Check for duplicate points or close points
    # The normalization step in _polygon_min_score handles scale.
    # But if points are identical, cross product is 0.
    
    score = _polygon_min_score(poly)
    
    # Penalty for non-convexity is handled by score being 0.
    # But Nelder-Mead needs a smooth-ish landscape.
    # Let's add a penalty if the angles are not well-distributed.
    # Ideally, gaps between consecutive angles (including wrap-around) should be roughly equal.
    # Gap i = angles[i+1] - angles[i]. Gap last = 2*pi + angles[0] - angles[-1].
    gaps = np.diff(angles)
    wrap_gap = 2 * np.pi + angles[0] - angles[-1]
    all_gaps = np.append(gaps, wrap_gap)
    
    # Penalize if any gap is too small (causes close vertices) or too large (non-uniform)
    # We want to maximize the minimum score.
    # If score is 0, we return a large positive value (since we minimize).
    if score == 0.0:
        # Additional penalty for bad geometry
        min_gap = np.min(all_gaps)
        return 10.0 - min_gap # High cost
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    n = 20
    # Regular polygon baseline
    regular_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = np.stack([np.cos(regular_angles), np.sin(regular_angles)], axis=1)
    best_score = _polygon_min_score(best_poly)
    
    # Optimization
    # Initial guess: regular angles
    x0 = regular_angles.copy()
    
    # We will use scipy.optimize.minimize
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback if scipy is not available (though prompt says it is)
        return best_poly

    # Nelder-Mead doesn't handle constraints well, so we rely on the penalty in objective.
    # However, sorting the angles inside objective makes it non-smooth.
    # To improve this, we can parameterize the gaps between angles.
    # Let g_i > 0 be the gap after vertex i. sum(g_i) = 2*pi.
    # This ensures convexity and cyclic order automatically.
    # There are 19 free parameters (since sum is fixed).
    
    def _objective_gaps(gaps):
        """
        gaps: array of 19 positive numbers.
        The 20th gap is determined by 2*pi - sum(gaps).
        We need all gaps > 0.
        """
        # If any gap <= 0, return high penalty
        if np.any(gaps <= 0):
            return 10.0
        gap_20 = 2.0 * np.pi - np.sum(gaps)
        if gap_20 <= 0:
            return 10.0
        
        all_gaps = np.append(gaps, gap_20)
        
        # Compute angles
        angles = np.cumsum(all_gaps)
        angles[0] = 0.0 # Fix rotation
        
        x = np.cos(angles)
        y = np.sin(angles)
        poly = np.stack([x, y], axis=1)
        
        score = _polygon_min_score(poly)
        
        if score == 0.0:
            # Penalize small gaps
            min_gap = np.min(all_gaps)
            return 10.0 - min_gap
        
        return -score

    # Initial gaps: regular
    regular_gap = 2.0 * np.pi / 20.0
    g0 = np.full(19, regular_gap)
    
    # Bounds for gaps: (epsilon, 2*pi - 19*epsilon)
    # We don't strictly need bounds for Nelder-Mead if penalty is high, but it helps.
    # Let's just run Nelder-Mead.
    
    best_gaps = g0.copy()
    current_score = best_score
    
    # Multiple restarts with different initial perturbations
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            g_start = g0.copy()
        else:
            # Perturb regular gaps
            perturbation = np.random.normal(0, 0.1, 19)
            g_start = g0 + perturbation
            # Ensure positive
            g_start = np.maximum(g_start, 0.1)
            # Ensure sum < 2*pi
            s = np.sum(g_start)
            if s >= 2.0 * np.pi:
                g_start = g_start * (1.9 * np.pi / s)
        
        try:
            res = minimize(_objective_gaps, g_start, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.success or res.fun < -current_score:
                g_opt = res.x
                # Verify the result
                gap_20 = 2.0 * np.pi - np.sum(g_opt)
                if gap_20 > 0 and np.all(g_opt > 0):
                    all_gaps = np.append(g_opt, gap_20)
                    angles = np.cumsum(all_gaps)
                    angles[0] = 0.0
                    x = np.cos(angles)
                    y = np.sin(angles)
                    poly = np.stack([x, y], axis=1)
                    
                    s = _polygon_min_score(poly)
                    if s > current_score:
                        current_score = s
                        best_poly = poly.copy()
                        best_gaps = g_opt.copy()
                        
        except Exception:
            continue
            
        if time.time() > deadline:
            break

    return best_poly

# EVOLVE-BLOCK-END
