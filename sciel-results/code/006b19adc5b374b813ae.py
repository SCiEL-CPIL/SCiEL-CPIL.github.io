# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_vertex_score(polygon, v_idx):
    """Compute the per-vertex score for vertex v_idx of polygon."""
    n = len(polygon)
    v = polygon[v_idx]
    # Distances to all other vertices
    diffs = polygon - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self
    dists[v_idx] = np.inf
    # Sort the 19 distances
    d = np.sort(dists)
    # Check if any < 0.05
    if d[0] < 0.05:
        return 0.0
    # Find k: first index minimizing d[k+1] - d[k]
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select 4 distances
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _polygon_min_score(polygon):
    """Compute the minimum vertex score over all vertices."""
    n = len(polygon)
    scores = [_compute_vertex_score(polygon, i) for i in range(n)]
    return min(scores)

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    dists = np.sqrt(np.sum(poly**2, axis=1))
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return poly
    return poly / mean_dist

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex with correct total turning."""
    n = len(polygon)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_prods = []
    for i in range(n):
        v1 = polygon[(i+1) % n] - polygon[i]
        v2 = polygon[(i+2) % n] - polygon[(i+1) % n]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    # Total turning should be ±2π
    # Compute angles of edges
    edge_angles = []
    for i in range(n):
        v = polygon[(i+1) % n] - polygon[i]
        edge_angles.append(np.arctan2(v[1], v[0]))
    # Compute signed turns
    turns = []
    for i in range(n):
        a1 = edge_angles[i]
        a2 = edge_angles[(i+1) % n]
        turn = a2 - a1
        # Normalize to (-π, π]
        while turn > np.pi:
            turn -= 2*np.pi
        while turn <= -np.pi:
            turn += 2*np.pi
        turns.append(turn)
    total_turn = sum(turns)
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return False
    return True

def _build_polygon_from_params(params, n=20):
    """Build polygon from radial modulation parameters.
    params: array of [a0, a1_cos, a1_sin, a2_cos, a2_sin, ...]
    r(θ) = a0 + sum_{m=1}^{M} (a_m*cos(m*θ) + b_m*sin(m*θ))
    """
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = np.zeros(n)
    r += params[0]  # a0
    idx = 1
    for m in range(1, 6):  # m = 1 to 5
        a_m = params[idx]
        b_m = params[idx+1]
        r += a_m * np.cos(m * theta) + b_m * np.sin(m * theta)
        idx += 2
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    polygon = np.stack([x, y], axis=1)
    return polygon

def _score_from_params(params, n=20):
    """Compute negative min vertex score (for minimization)."""
    polygon = _build_polygon_from_params(params, n)
    polygon = _normalize(polygon)
    if not _is_strictly_convex(polygon):
        return 10.0  # large penalty
    score = _polygon_min_score(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n=20 vertices that tries to maximize the minimum
    per-vertex equidistance score. Uses radial modulation with Fourier modes and
    coordinate descent / Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    n = 20
    
    # Start with regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    best_polygon = np.stack([x, y], axis=1)
    best_score = _polygon_min_score(_normalize(best_polygon))
    
    # Parameters: a0, then for m=1..5: a_m, b_m
    # Start near regular: a0=1, all others 0
    n_params = 1 + 2*5  # a0 + 5 pairs
    best_params = np.zeros(n_params)
    best_params[0] = 1.0
    
    # Try different initializations and optimize
    # First, try a simple coordinate descent on the modulation amplitudes
    
    # Initialize with small random perturbations
    np.random.seed(42)
    
    # Try multiple random starts
    n_starts = 8
    for start_idx in range(n_starts):
        if time.time() > deadline - 60.0:
            break
        
        if start_idx == 0:
            params = best_params.copy()
        else:
            params = best_params.copy()
            # Add random perturbations
            for i in range(1, n_params):
                params[i] += np.random.uniform(-0.1, 0.1)
        
        # Use Nelder-Mead to optimize
        try:
            result = minimize(
                _score_from_params,
                params,
                method='Nelder-Mead',
                args=(n,),
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() > deadline - 30.0:
                break
            if result.fun < -best_score:
                best_score = -result.fun
                best_params = result.x.copy()
                best_polygon = _normalize(_build_polygon_from_params(best_params, n))
        except Exception:
            pass
    
    # Coordinate descent polish on individual vertex radii
    # This is a more direct approach: adjust each vertex's radial distance
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    
    # First, try to optimize radii directly
    def _score_from_radii(radii):
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)
        polygon = np.stack([x, y], axis=1)
        polygon = _normalize(polygon)
        if not _is_strictly_convex(polygon):
            return 10.0
        return -_polygon_min_score(polygon)
    
    # Start from the best polygon found
    # Extract radii from best_polygon
    bp = best_polygon
    bp_radii = np.sqrt(np.sum(bp**2, axis=1))
    current_radii = bp_radii.copy()
    current_score = -_score_from_radii(current_radii)
    if current_score < -best_score:
        best_score = -current_score
        best_polygon = _normalize(np.stack([current_radii*np.cos(theta), current_radii*np.sin(theta)], axis=1))
    
    # Coordinate descent: adjust each radius one at a time
    for outer_iter in range(30):
        if time.time() > deadline - 60.0:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 30.0:
                break
            # Try small adjustments to radii[i]
            best_local = current_radii[i]
            best_local_score = -_score_from_radii(current_radii)
            for delta in [0.05, -0.05, 0.1, -0.1, 0.02, -0.02, 0.01, -0.01]:
                trial = current_radii.copy()
                trial[i] = max(0.1, current_radii[i] + delta)
                trial_score = -_score_from_radii(trial)
                if trial_score < best_local_score:
                    best_local_score = trial_score
                    best_local = trial[i]
            if best_local != current_radii[i]:
                current_radii[i] = best_local
                improved = True
        if not improved:
            break
        # Update best
        trial_score = -_score_from_radii(current_radii)
        if trial_score < -best_score:
            best_score = -trial_score
            best_polygon = _normalize(np.stack([current_radii*np.cos(theta), current_radii*np.sin(theta)], axis=1))
    
    # Final verification
    final_polygon = _normalize(best_polygon)
    if not _is_strictly_convex(final_polygon):
        # Fallback to regular polygon
        final_polygon = _normalize(np.stack([np.cos(theta), np.sin(theta)], axis=1))
    
    return final_polygon

# EVOLVE-BLOCK-END
