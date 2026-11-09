# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_vertex_scores(points):
    """Compute per-vertex equidistance scores for a polygon."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(dists[i])  # d_0 <= ... <= d_18
        # Check if any distance < 0.05
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d)  # d_{k+1} - d_k for k=0..17
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        if D < 0.1:
            scores[i] = 0.0
            continue
        # Take four distances
        if k == 0:
            four_d = d[0:4]
        elif k == 17:
            four_d = d[15:19]
        else:
            four_d = d[k-1:k+3]
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return scores

def _polygon_score(points):
    """Compute the overall polygon score (min vertex score)."""
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    if not _is_valid_convex(normalized):
        return 0.0
    
    scores = _compute_vertex_scores(normalized)
    return np.min(scores)

def _is_valid_convex(points):
    """Check if polygon is strictly convex with nonzero turns of one sign."""
    n = len(points)
    if n < 3:
        return False
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return False
    # Check turns
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return False
    # Total turning should be +-2*pi (for a simple convex polygon, this is automatic)
    return True

def _angles_to_points(angles):
    """Convert angles to points on unit circle."""
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _objective(angles_flat):
    """Objective function: negative of polygon score."""
    angles = angles_flat.copy()
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    points = _angles_to_points(angles)
    score = _polygon_score(points)
    return -score

def _random_angles(rng, n=20):
    """Generate random initial angles for a convex polygon."""
    # Generate random gaps that sum to 2*pi
    gaps = rng.random(n) + 0.1  # Ensure positive
    gaps = gaps / np.sum(gaps) * 2 * np.pi
    angles = np.cumsum(gaps)
    angles[-1] = 2 * np.pi  # Ensure we close the loop
    angles = angles[:-1]  # Remove the last (redundant) angle
    return angles

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n = 20
    rng = np.random.RandomState(42)
    
    best_score = 0.0
    best_points = None
    
    # Start with regular polygon as initial best
    regular_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    regular_points = _angles_to_points(regular_angles)
    best_score = _polygon_score(regular_points)
    best_points = regular_points
    
    # Try multiple random restarts with Nelder-Mead optimization
    num_restarts = 50
    max_iter_per_restart = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial angles
        angles = _random_angles(rng, n)
        
        # Optimize
        try:
            result = minimize(
                _objective,
                angles,
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_restart, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            opt_angles = np.mod(result.x, 2 * np.pi)
            opt_points = _angles_to_points(opt_angles)
            opt_score = _polygon_score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        
        except Exception:
            continue
    
    # Final polish: try local refinement on best points
    if best_points is not None and time.time() < deadline:
        # Convert best points back to angles
        angles = np.arctan2(best_points[:, 1], best_points[:, 0])
        angles = np.mod(angles, 2 * np.pi)
        
        # Try a few more local optimizations with smaller steps
        for polish_iter in range(5):
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _objective,
                    angles,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                opt_angles = np.mod(result.x, 2 * np.pi)
                opt_points = _angles_to_points(opt_angles)
                opt_score = _polygon_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    angles = opt_angles
            except Exception:
                break
    
    # Normalize the best points
    centroid = np.mean(best_points, axis=0)
    centered = best_points - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        best_points = centered / mean_dist
    
    return best_points

# EVOLVE-BLOCK-END
