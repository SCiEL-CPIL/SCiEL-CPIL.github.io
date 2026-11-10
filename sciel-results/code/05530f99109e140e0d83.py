# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms

def _normalize_sphere(points: np.ndarray) -> np.ndarray:
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _generate_symmetric_candidates(ico: np.ndarray, rng: np.random.RandomState) -> list:
    """
    Generate a diverse set of initial candidates by placing 2 extra points
    in symmetric positions relative to the icosahedron.
    """
    candidates = []
    
    # 1. Poles (0,0,1) and (0,0,-1)
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    cand = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    candidates.append(_normalize_sphere(cand))
    
    # 2. Points along the 5-fold axes (z-axis is one, but ico vertices are already there? No, ico vertices are (0,1,phi) etc)
    # The 5-fold axes of the icosahedron pass through the centers of opposite faces.
    # One such axis is the z-axis? No, for the standard ico vertices given above:
    # Faces are triangles. The centers of the faces are at distance 1 from origin?
    # Let's calculate face centers.
    # Actually, it is known that adding the 2 poles (0,0,1) and (0,0,-1) to the icosahedron
    # creates a configuration with D5h symmetry (if ico is aligned properly).
    # Wait, the standard ico vertices given are NOT aligned with the z-axis as a 5-fold axis.
    # The vertices are (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1).
    # The 5-fold axes pass through (0,0,1)? No.
    # The vector (0,0,1) is perpendicular to (0,1,phi)? Dot product is phi. Not 0.
    # So (0,0,1) is NOT a symmetry axis of this specific ico orientation.
    
    # Let's find the actual 5-fold axis.
    # Sum of all 12 vertices is 0.
    # The 5-fold axis goes through the center of a face.
    # Face 1: v0=(0,1,phi), v1=(0,-1,phi), v2=(1,phi,0)? No, need to check distances.
    # Dist v0-v1 = 2. Dist v0-v2 = sqrt(0+ (phi-1)^2 + phi^2) = sqrt( (2phi-1)^2 + phi^2 )?
    # phi^2 = phi+1.
    # (phi-1) = 1/phi.
    # dist^2 = 1/phi^2 + phi^2 = 1/(phi+1) + phi+1 = (1 + (phi+1)^2)/(phi+1) = (1 + phi^2+2phi+1)/(phi+1) = (2+phi+2phi+1)/(phi+1) = (3+3phi)/(phi+1) = 3.
    # So edge length is sqrt(2) for (0,1,phi)-(0,-1,phi).
    # Let's just use the fact that the optimal 14-point set is likely related to the icosahedron.
    
    # Alternative: Random restarts with symmetry constraints.
    # Generate random pairs of points, then reflect them to ensure some symmetry?
    # Or just random points on the sphere.
    
    # Let's add random perturbations of the pole configuration
    for _ in range(20):
        noise = rng.randn(14, 3) * 0.2
        cand = candidates[0] + noise
        candidates.append(_normalize_sphere(cand))
        
    # Pure random restarts
    for _ in range(20):
        cand = rng.randn(14, 3)
        candidates.append(_normalize_sphere(cand))
        
    # "Spherical Code" heuristics: 
    # Try to place points to maximize minimum distance (Tammes problem).
    # We can use a greedy construction:
    # Start with 1 point. Add next point farthest from existing.
    # This is O(N^2) per step, cheap for N=14.
    greedy_pts = np.zeros((14, 3))
    greedy_pts[0] = rng.randn(3)
    greedy_pts[0] /= np.linalg.norm(greedy_pts[0])
    for i in range(1, 14):
        best_pos = None
        best_min_dist = -1
        # Sample 1000 random points
        samples = rng.randn(1000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        # Compute min distance from each sample to existing points
        # Existing: greedy_pts[:i]
        for s in samples:
            dists = np.linalg.norm(greedy_pts[:i] - s, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = s
        greedy_pts[i] = best_pos
    candidates.append(greedy_pts)
    
    return candidates

def _refine_nelder_mead(points: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Refine points using Nelder-Mead on the raw metric.
    Nelder-Mead does not require gradients and handles the non-smoothness of min/max better.
    """
    start_time = time.time()
    best_metric = _evaluate(points)
    best_points = points.copy()
    
    # Nelder-Mead objective: negative metric
    def objective(params):
        pts = params.reshape(14, 3)
        # Normalize to sphere to keep scale fixed (metric is scale invariant, but NM might drift)
        # Actually, if we normalize inside, the landscape is on the sphere.
        # But NM works in Euclidean space.
        # Let's just evaluate on the raw points. If they drift off sphere, the metric is still valid.
        # However, to prevent explosion, we can penalize large norms or normalize.
        # Normalizing inside makes the function periodic/discontinuous at the origin.
        # Better to constrain to sphere? No, NM doesn't do constraints.
        # Let's just evaluate. The metric is scale invariant, so NM will likely find a local min in the "basin" of the sphere.
        # To help, we can normalize the points before evaluation.
        pts_norm = _normalize_sphere(pts)
        return -_evaluate(pts_norm)
        
    # Initial simplex
    # Perturb initial points slightly to form simplex?
    # Standard NM uses a simplex of size N+1.
    # We can just let scipy handle it with initial guess.
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        refined_pts = result.x.reshape(14, 3)
        refined_pts = _normalize_sphere(refined_pts)
        refined_metric = _evaluate(refined_pts)
        
        if refined_metric > best_metric:
            best_points = refined_pts
            best_metric = refined_metric
            
    except Exception:
        pass
        
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    rng = np.random.RandomState(42)
    ico = _icosahedron_vertices()
    
    candidates = _generate_symmetric_candidates(ico, rng)
    
    best_points = None
    best_metric = 0.0
    
    # Evaluate all initial candidates
    for cand in candidates:
        metric = _evaluate(cand)
        if metric > best_metric:
            best_metric = metric
            best_points = cand.copy()
            
    # Refine top candidates
    # Sort candidates by metric to refine the best ones first
    scored_candidates = []
    for cand in candidates:
        m = _evaluate(cand)
        scored_candidates.append((m, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top 10 candidates
    for i, (m, cand) in enumerate(scored_candidates[:10]):
        if time.time() > deadline - 60:
            break
            
        refined = _refine_nelder_mead(cand, 30)
        refined_metric = _evaluate(refined)
        
        if refined_metric > best_metric:
            best_metric = refined_metric
            best_points = refined.copy()
            
    # If time remains, do a second pass of refinement on the best point found
    if best_points is not None and time.time() < deadline - 30:
        # Try a few random restarts from the best point
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            noise = rng.randn(14, 3) * 0.1
            perturbed = best_points + noise
            perturbed = _normalize_sphere(perturbed)
            
            refined = _refine_nelder_mead(perturbed, 20)
            refined_metric = _evaluate(refined)
            
            if refined_metric > best_metric:
                best_metric = refined_metric
                best_points = refined.copy()
                
    # Final fallback
    if best_points is None:
        best_points = ico
        
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.vstack([ico, np.array([[0, 0, 1]]), np.array([[0, 0, -1]])])
        best_points = _normalize_sphere(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
