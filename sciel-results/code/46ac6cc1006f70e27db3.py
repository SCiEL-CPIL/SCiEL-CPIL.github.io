# EVOLVE-BLOCK-START
import numpy as np
import math

def _vertex_score(distances: np.ndarray) -> float:
    """Compute the score for a single vertex given sorted distances to other 19 vertices."""
    d = np.sort(distances)
    if d[0] < 0.05:
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _min_vertex_score(points: np.ndarray) -> float:
    """Compute the minimum vertex score over all vertices."""
    n = len(points)
    scores = []
    for i in range(n):
        diffs = points - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.delete(dists, i)  # exclude self
        scores.append(_vertex_score(dists))
    return min(scores)

def _is_strictly_convex(points: np.ndarray) -> bool:
    """Check if the polygon is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    cross_signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross == 0:
            return False
        cross_signs.append(np.sign(cross))
    # All cross products must have the same sign
    if len(set(cross_signs)) > 1:
        return False
    return True

def _normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return None
    points = points / mean_dist
    return points

def _evaluate_params(params: np.ndarray) -> float:
    """
    Evaluate the score for a given set of parameters.
    params: [theta_0, ..., theta_19, r_0, ..., r_19]
    """
    n = 20
    thetas = params[:n]
    radii = params[n:]
    
    # Sort by angle to maintain cyclic order
    # Since we perturb small amounts, sorting might change indices if we cross boundaries, 
    # but for a convex polygon, the order of vertices is fixed by angle.
    # However, if we allow large perturbations, the "cyclic order" of the input array 
    # might not match the angular order. 
    # The task requires returning points in cyclic order.
    # So we must sort the vertices by angle before constructing the polygon.
    # But wait, if we sort by angle, the "vertex i" in the score calculation 
    # corresponds to the i-th smallest angle. This is fine as long as we are consistent.
    
    # Ensure angles are in [0, 2pi)
    thetas = np.sort(thetas) % (2.0 * np.pi)
    thetas = np.sort(thetas)
    
    # Check for duplicate angles (degenerate)
    if np.min(np.diff(np.concatenate([thetas, [thetas[0] + 2.0 * np.pi]]))) < 1e-6:
        return 0.0
        
    # Sort radii to match sorted thetas? 
    # No, the parameterization is tied to the index. 
    # If we sort thetas, we must also sort the corresponding radii.
    # This effectively re-labels the vertices. Since the polygon is undirected 
    # and the score is symmetric over vertices (min over all), re-labeling is fine.
    # However, the "cyclic order" is determined by the sorted angles.
    
    sort_idx = np.argsort(thetas)
    thetas = thetas[sort_idx]
    radii = radii[sort_idx]
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    points = np.stack([x, y], axis=1)
    
    normalized_points = _normalize(points)
    if normalized_points is None:
        return 0.0
        
    if not _is_strictly_convex(normalized_points):
        return 0.0
        
    return _min_vertex_score(normalized_points)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon using Simulated Annealing on angles and radii.
    """
    n = 20
    # Initial parameters: regular polygon
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    params = np.concatenate([thetas, radii])
    
    current_score = _evaluate_params(params)
    best_params = params.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.1
    T_min = 1e-6
    alpha = 0.995
    max_iter = 20000
    
    np.random.seed(42)
    
    for i in range(max_iter):
        # Perturb
        new_params = params + np.random.normal(0, 0.05, 2 * n)
        # Keep radii positive
        new_params[n:] = np.abs(new_params[n:]) + 0.1
        
        new_score = _evaluate_params(new_params)
        
        # Acceptance criterion
        if new_score > current_score or np.random.random() < math.exp((new_score - current_score) / T):
            current_score = new_score
            params = new_params
            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
        
        T *= alpha
        if T < T_min:
            break
            
    # Reconstruct best points
    thetas = best_params[:n]
    radii = best_params[n:]
    
    thetas = np.sort(thetas) % (2.0 * np.pi)
    thetas = np.sort(thetas)
    
    sort_idx = np.argsort(thetas)
    thetas = thetas[sort_idx]
    radii = radii[sort_idx]
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    points = np.stack([x, y], axis=1)
    
    normalized_points = _normalize(points)
    if normalized_points is None:
        # Fallback to regular polygon
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(base_angles)
        y = np.sin(base_angles)
        points = np.stack([x, y], axis=1)
        normalized_points = _normalize(points)
        
    return normalized_points

# EVOLVE-BLOCK-END
