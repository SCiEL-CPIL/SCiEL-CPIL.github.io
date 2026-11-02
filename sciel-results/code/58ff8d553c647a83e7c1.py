"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools


def _all_triangle_areas(pts):
    """Compute all triangle areas among triples of points, vectorized."""
    n = len(pts)
    if n < 3:
        return np.array([])
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas.append(0.5 * cross)
    
    if len(areas) == 0:
        return np.array([])
    return np.concatenate(areas)


def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _compute_gram_matrix(pts):
    """
    Compute the Gram matrix of squared distances.
    G[i,j] = ||p_i - p_j||^2
    """
    n = len(pts)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = pts[i] - pts[j]
            G[i, j] = np.dot(diff, diff)
    return G


def _triangle_area_from_gram(G, i, j, k):
    """
    Compute triangle area from Gram matrix entries.
    Area = 0.5 * sqrt(2*(G[i,j]*G[j,k]*G[k,i]) - G[i,j]^2*G[j,k] - G[j,k]^2*G[k,i] - G[k,i]^2*G[i,j])
    """
    a2 = G[i, j]
    b2 = G[j, k]
    c2 = G[k, i]
    
    # Heron's formula variant
    val = 2.0 * (a2 * b2 * c2) - a2**2 * b2 - b2**2 * c2 - c2**2 * a2
    if val < 0:
        val = 0.0
    return 0.25 * np.sqrt(val)


def _neg_sdp_score(params):
    """
    Negative score based on SDP relaxation.
    params encodes:
    - First 2*N elements: point coordinates (for reconstruction)
    - We use a penalty-based approach to enforce constraints
    
    This is a simplified SDP-inspired approach using coordinate optimization
    with area constraints as soft penalties.
    """
    pts = params.reshape(-1, 2)
    
    if len(pts) < 3:
        return 0.0
    
    # Compute all triangle areas
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    # Normalize: we want min_area / hull_area to be maximized
    # Equivalent to minimizing -min_area / hull_area
    
    # Use a penalty for small areas to encourage all triangles to be large
    min_area = np.min(areas)
    
    # Penalize areas that are much smaller than the minimum
    # This creates a smoother landscape
    threshold = min_area * 0.9
    violations = np.maximum(0, threshold - areas)
    penalty = np.sum(violations**2)
    
    score = min_area / hull_area - 0.01 * penalty / hull_area
    return -score


def _neg_smooth_score(params, temperature=0.01):
    """Negative smooth score for minimization."""
    pts = params.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    # Soft min: -T * log(sum(exp(-areas/T)))
    min_val = np.min(areas)
    exp_terms = np.exp(-(areas - min_val) / temperature)
    soft_min = min_val - temperature * np.log(np.sum(exp_terms))
    
    return -soft_min / hull_area


def _neg_score(params):
    """Negative score for minimization (hard min)."""
    pts = params.reshape(-1, 2)
    return -_score_points(pts)


def _sdp_relaxation_init():
    """
    Generate initial configuration using SDP-inspired principles.
    The key insight: in the optimal configuration, all triangle areas
    should be as equal as possible (by the max-min principle).
    
    This suggests a configuration where points are distributed to
    equalize triangle areas, which is related to optimal experimental design.
    """
    inits = []
    
    # Start with regular polygon (known to be near-optimal for small n)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts.copy())
    
    # Try configurations that promote area equalization
    # Use optimization to find a configuration where all triangle areas are similar
    
    return inits


def _equalize_areas(pts, max_iter=100, tol=1e-8):
    """
    Iteratively adjust points to equalize triangle areas.
    This is inspired by the SDP relaxation where we want all areas >= t
    and we maximize t.
    """
    pts = pts.copy()
    
    for iteration in range(max_iter):
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            break
        
        min_area = np.min(areas)
        max_area = np.max(areas)
        
        if max_area - min_area < tol:
            break
        
        # Find the triangle with minimum area
        min_idx = np.argmin(areas)
        
        # Identify which triangle this is
        # We need to map index back to (i,j,k)
        # For simplicity, use gradient-free adjustment
        # Push points apart slightly
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[j] - pts[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                # Repulsive force
                force = diff / dist * 0.001
                pts[i] -= force
                pts[j] += force
        
        # Keep points in reasonable bounds
        pts = np.clip(pts, -2, 2)
        
        # Normalize to unit area
        hull_area = _hull_area(pts)
        if hull_area > 1e-12:
            scale = np.sqrt(1.0 / hull_area)
            pts = pts * scale
    
    return pts


def _sdp_inspired_optimize(pts, deadline):
    """
    SDP-inspired optimization:
    1. Start from a good initial configuration
    2. Use the insight that optimal configs equalize triangle areas
    3. Perform local search with area-equalization objective
    
    The SDP relaxation suggests that the optimal t (minimum area) is achieved
    when as many triangles as possible have area exactly t, creating a
    "tight" configuration.
    """
    best_score = _score_points(pts)
    best_pts = pts.copy()
    
    # Phase 1: Area equalization
    equalized = _equalize_areas(pts, max_iter=50)
    score_eq = _score_points(equalized)
    if score_eq > best_score:
        best_score = score_eq
        best_pts = equalized.copy()
    
    # Phase 2: Local refinement with soft-min
    if time.time() < deadline - 5:
        for temp in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 3:
                break
            
            try:
                result = minimize(
                    lambda x: _neg_smooth_score(x, temperature=temp),
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-2, 2)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if result.fun is not None:
                    candidate_pts = result.x.reshape(-1, 2)
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead refinement
    if time.time() < deadline - 2:
        try:
            result = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            
            if result.fun is not None:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception:
            pass
    
    return best_pts, best_score


def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts.copy())
    
    # 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center])
    inits.append(pts.copy())
    
    # Two concentric polygons (6+7)
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(np.vstack([pts1, pts2]))
    
    # Three rings (4+4+5)
    angles1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts1 = np.column_stack([0.33 * np.cos(angles1), 0.33 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    pts2 = np.column_stack([0.66 * np.cos(angles2), 0.66 * np.sin(angles2)])
    angles3 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
    inits.append(np.vstack([pts1, pts2, pts3]))
    
    # Random perturbations of regular polygon
    np.random.seed(42)
    for seed in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + np.random.normal(0, 0.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts.copy())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Generate diverse initializations
    inits = _generate_initializations()
    
    # Evaluate all initializations
    for pts in inits:
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # SDP-inspired optimization from multiple starting points
    if best_pts is not None:
        if time.time() < deadline - 30:
            # Try from the best initial point
            optimized_pts, optimized_score = _sdp_inspired_optimize(best_pts, deadline - 10)
            if optimized_score > best_score:
                best_score = optimized_score
                best_pts = optimized_pts
        
        # Try from a few other good starting points
        if time.time() < deadline - 20:
            for init_idx in [0, 1, 2]:
                if time.time() > deadline - 10:
                    break
                
                if init_idx < len(inits):
                    try:
                        optimized_pts, optimized_score = _sdp_inspired_optimize(inits[init_idx], deadline - 5)
                        if optimized_score > best_score:
                            best_score = optimized_score
                            best_pts = optimized_pts
                    except Exception:
                        pass
    
    # Final local refinement
    if best_pts is not None and time.time() < deadline - 5:
        for i in range(2):
            if time.time() > deadline - 3:
                break
            
            start = best_pts.copy()
            if i > 0:
                start = start + np.random.normal(0, 0.02, size=start.shape)
            
            try:
                result = minimize(
                    _neg_score,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if result.fun is not None:
                    candidate_pts = result.x.reshape(-1, 2)
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        
            except Exception:
                pass
    
    # Return the best points found
    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        # Fallback: return first initialization
        return inits[0]
# EVOLVE-BLOCK-END
