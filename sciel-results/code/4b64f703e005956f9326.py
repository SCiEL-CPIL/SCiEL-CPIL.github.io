import numpy as np
import time
from scipy.optimize import minimize

def _normalize(points):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist

def _vertex_scores(points):
    """
    Compute the score for each vertex of a normalized polygon.
    Returns an array of 20 scores.
    """
    n = points.shape[0]
    # Compute all pairwise distances
    # dists[i, j] = distance from point i to point j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        # Distances from vertex i to all others, excluding self
        d = dists[i].copy()
        d[i] = np.inf
        sorted_d = np.sort(d)
        
        # Check: any dist < 0.05
        if sorted_d[0] < 0.05:
            scores[i] = 0.0
            continue
            
        # Find smallest gap
        gaps = sorted_d[1:] - sorted_d[:-1]
        k = np.argmin(gaps)
        D = (sorted_d[k] + sorted_d[k+1]) / 2.0
        
        # Check: D < 0.1
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            d_sel = sorted_d[0:4]
        elif k == 17:
            d_sel = sorted_d[15:19]
        else:
            d_sel = sorted_d[k-1:k+3]
            
        # Calculate raw score
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        if raw == 0:
            scores[i] = 0.0
        else:
            scores[i] = min(1.0, 1.0 / raw)
            
    return scores

def _min_vertex_score(points):
    """Return the minimum vertex score."""
    return np.min(_vertex_scores(points))

def _fourier_to_points(coeffs):
    """
    Convert Fourier coefficients to polygon points.
    coeffs: array of length 2*max_mode + 1? 
    Let's use a simpler parameterization:
    angles[i] = 2*pi*i/n + sum_{m=1}^{M} (a_m * cos(2*pi*m*i/n) + b_m * sin(2*pi*m*i/n))
    
    We restrict to low frequency modes to ensure smoothness.
    coeffs = [a_1, b_1, a_2, b_2, ...]
    """
    n = 20
    i = np.arange(n)
    base_angles = 2 * np.pi * i / n
    
    # Parse coefficients
    # Let's say we use modes 1, 2, 3.
    # coeffs: [a1, b1, a2, b2, a3, b3]
    if len(coeffs) < 6:
        # Pad with zeros
        coeffs = np.pad(coeffs, (0, 6 - len(coeffs)))
        
    a1, b1, a2, b2, a3, b3 = coeffs[:6]
    
    angles = base_angles.copy()
    angles += a1 * np.cos(2*np.pi*1*i/n) + b1 * np.sin(2*np.pi*1*i/n)
    angles += a2 * np.cos(2*np.pi*2*i/n) + b2 * np.sin(2*np.pi*2*i/n)
    angles += a3 * np.cos(2*np.pi*3*i/n) + b3 * np.sin(2*np.pi*3*i/n)
    
    # Ensure angles are sorted to maintain cyclic order?
    # The Fourier series with small coefficients should keep the order.
    # However, if coefficients are too large, the order might swap.
    # The task requires "cyclic order". 
    # If we just take cos/sin of the angles, the points will be in the order of the angles.
    # If the angles are not monotonic, the polygon might self-intersect or not be in cyclic order.
    # To be safe, we can sort the angles? But that changes the mapping from coeffs to points.
    # Actually, the task says "returning a (20, 2) array of finite vertex coordinates in cyclic order".
    # It does NOT say the input parameters must map to cyclic order. It says the OUTPUT must be in cyclic order.
    # So we can generate points, then sort them by angle to ensure cyclic order.
    # BUT, sorting by angle might change the vertex indices, which affects the score calculation?
    # The score is invariant to rotation but depends on the set of points.
    # Wait, the vertex score depends on the distances from a specific vertex to others.
    # If we sort, we are just re-labeling the vertices. The MINIMUM score over all vertices is invariant to relabeling.
    # So sorting is safe for the metric.
    
    # However, if the angles cross (non-monotonic), the polygon is not strictly convex in the sense of the original angle sequence.
    # But if we sort the angles, we get a convex polygon (since they are on the circle).
    # The task requires the RETURNED array to be in cyclic order.
    # So:
    # 1. Generate angles.
    # 2. Wrap to [0, 2*pi).
    # 3. Sort angles.
    # 4. Compute points.
    
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    return points

def _objective(coeffs):
    """
    Objective function for optimization.
    Returns negative of the minimum vertex score.
    """
    points = _fourier_to_points(coeffs)
    points_norm = _normalize(points)
    
    # Check convexity?
    # Since points are on a circle and sorted by angle, they form a convex polygon.
    # Strict convexity requires no repeated points and nonzero turns.
    # If any two angles are identical, points are repeated -> invalid.
    # We should add a penalty for small gaps between sorted angles.
    
    # Calculate gaps between consecutive angles (cyclic)
    gaps = np.diff(angles)
    gaps = np.append(gaps, angles[0] + 2*np.pi - angles[-1])
    min_gap = np.min(gaps)
    
    # If min_gap is too small, points are too close, potentially invalid or low score.
    # The vertex score will naturally be low if points are close.
    # But we need to ensure validity (no repeated points).
    # If min_gap < 1e-6, consider it invalid.
    if min_gap < 1e-6:
        return 1.0  # High penalty (we are minimizing, so high value is bad)
    
    score = _min_vertex_score(points_norm)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep 10s margin
    
    best_score = -np.inf
    best_coeffs = np.zeros(6)
    best_points = _fourier_to_points(best_coeffs)
    
    # Initial evaluation
    init_score = -_objective(best_coeffs)
    if init_score > best_score:
        best_score = init_score
        best_points = _fourier_to_points(best_coeffs)
        
    # Define bounds for Fourier coefficients
    # Angles are spaced by 2*pi/20 ~ 0.314.
    # Perturbations should be smaller than half the spacing to avoid overlap.
    # Mode 1 is a shift/ellipse. Mode 2 is a quadrilateral distortion.
    # Let's use bounds [-0.1, 0.1] for all coefficients initially.
    bounds = [(-0.1, 0.1)] * 6
    
    # Optimization Strategy:
    # 1. Multi-start L-BFGS-B.
    # 2. Use random restarts to explore the 6D space.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            x0 = np.zeros(6)
        else:
            # Random start within bounds
            x0 = rng.uniform(-0.1, 0.1, size=6)
            
        try:
            res = minimize(
                _objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_coeffs = res.x
                best_points = _fourier_to_points(best_coeffs)
                
        except Exception:
            pass
            
    # Final polish with tighter bounds if needed?
    # The current bounds are small.
    
    # Return the points
    return best_points

