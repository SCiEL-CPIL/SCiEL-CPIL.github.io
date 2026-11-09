# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(vertices, i):
    """
    Compute the score for vertex i of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    """
    n = len(vertices)
    rotated = np.roll(vertices, -i, axis=0)
    diffs = rotated - rotated[0]
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
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        v_score = _vertex_score(normalized, i)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _polygon_from_angle_perturbations(deltas):
    """
    Construct a convex polygon from angle perturbations.
    Base angles are 2*pi*i/20.
    deltas: (20,) array of perturbations.
    To ensure cyclic order and convexity, we restrict deltas to be small
    and ensure the cumulative angle is monotonic.
    Actually, just adding small deltas to regular polygon angles keeps it convex
    if deltas are small enough.
    """
    n = 20
    base_angles = 2 * np.pi * np.arange(n) / n
    angles = base_angles + deltas
    
    # Sort angles to ensure cyclic order? 
    # If deltas are small, order is preserved. 
    # If we allow larger deltas, we might need to sort, but then it's not a simple perturbation.
    # Let's assume deltas are small enough to preserve order (|deltas| < pi/10).
    
    # Construct vertices on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    
    return np.stack([x, y], axis=1)

def _objective_angle_perturbations(deltas):
    """
    Objective function for angle perturbation optimization.
    We want to maximize the minimum vertex score, so we minimize its negative.
    """
    try:
        poly = _polygon_from_angle_perturbations(deltas)
        score = _polygon_min_score(poly)
        return -score
    except Exception:
        return 1.0

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon
    best_poly = _polygon_from_angle_perturbations(np.zeros(n))
    best_score = _polygon_min_score(best_poly)
    best_deltas = np.zeros(n)
    
    # Optimization using Nelder-Mead (robust for non-smooth min-score landscape)
    # We can also try CMA-ES if available, but scipy.optimize.minimize with NM is standard.
    # Let's use a multi-start strategy with NM.
    
    bounds_delta = 0.2 # Max perturbation in radians (~11 degrees)
    
    def _nm_optimize(x0, maxiter=500):
        try:
            result = minimize(
                _objective_angle_perturbations, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.success or result.fun < -best_score:
                return result.x, -result.fun
        except Exception:
            pass
        return x0, -best_score
        
    # Start 0: Regular
    current_deltas, current_score = _nm_optimize(np.zeros(n))
    if current_score > best_score:
        best_score = current_score
        best_deltas = current_deltas
        best_poly = _polygon_from_angle_perturbations(best_deltas)

    # Multi-start with random perturbations
    np.random.seed(42)
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
            
        # Generate random initial deltas
        # We want to explore shapes that might "spread" vertices more evenly in terms of distance
        # Random small perturbations
        x0 = np.random.normal(0, 0.05, size=n)
        
        # Also try some structured perturbations
        # e.g. alternating signs
        if i % 2 == 0:
            x0 = np.random.normal(0, 0.1, size=n)
            # Enforce some symmetry? No, let's keep it general.
            
        # Run optimization
        candidate_deltas, candidate_score = _nm_optimize(x0, maxiter=300)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_deltas = candidate_deltas
            best_poly = _polygon_from_angle_perturbations(best_deltas)
            
    # Final polish with more iterations
    if time.time() < deadline - 30.0:
        final_deltas, final_score = _nm_optimize(best_deltas, maxiter=1000)
        if final_score > best_score:
            best_score = final_score
            best_deltas = final_deltas
            best_poly = _polygon_from_angle_perturbations(best_deltas)

    return best_poly

# EVOLVE-BLOCK-END
