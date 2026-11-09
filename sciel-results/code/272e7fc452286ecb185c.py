# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_score(points, beta=100.0):
    """
    Compute a smooth approximation of (d_min / d_max)^2.
    Uses soft-min and soft-max to create a differentiable objective.
    """
    dists = pdist(points, metric='euclidean')
    if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
        return 0.0
    
    # Soft min and max
    d_min_soft = -np.logsumexp(-beta * dists) / beta
    d_max_soft = np.logsumexp(beta * dists) / beta
    
    if d_max_soft <= 1e-10:
        return 0.0
        
    ratio = d_min_soft / d_max_soft
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # --- Phase 1: Symmetry-Reduced Optimization ---
    # We exploit the T_d (tetrahedral) symmetry group which has 12 rotations.
    # A fundamental domain of size 14 can be decomposed into orbits.
    # Common decomposition for 14 points in T_d:
    # - 1 orbit of size 1 (center? No, T_d doesn't fix a generic point to size 1 unless it's origin)
    # - Actually, T_d orbits have sizes 1, 4, 6, 12, 24...
    # - 14 = 1 + 4 + 4 + 5? No.
    # - 14 = 4 + 4 + 6? (Two orbits of 4, one of 6)
    # - 14 = 12 + 2? (One orbit of 12, one of 2? Orbit of 2 is not standard in T_d unless on axis)
    # Let's use a generic 14-point configuration that *can* have high symmetry, 
    # but instead of hard constraints, we optimize a "near-symmetric" structure 
    # or just use the symmetry to reduce parameters for the initial guess.
    
    # A better structural approach: 
    # The optimal packing of 14 points in 3D is likely related to a subset of 
    # vertices of a higher-symmetry polytope or a layered structure.
    # Let's try a specific T_d symmetric configuration:
    # Orbit 1: 4 points (vertices of a tetrahedron) - radius r1
    # Orbit 2: 4 points (vertices of an inverted tetrahedron) - radius r2
    # Orbit 3: 6 points (vertices of an octahedron) - radius r3
    # Total = 14 points.
    
    # Define the base vectors for these orbits (normalized)
    # Tetrahedron vertices (normalized):
    t1 = np.array([1, 1, 1]) / np.sqrt(3)
    t2 = np.array([1, -1, -1]) / np.sqrt(3)
    t3 = np.array([-1, 1, -1]) / np.sqrt(3)
    t4 = np.array([-1, -1, 1]) / np.sqrt(3)
    
    # Inverted Tetrahedron (same as above, just scaled differently if needed, but geometrically same set)
    # To distinguish orbits, we might need different angular offsets or just different radii.
    # If we use the same 4 vectors for both orbits, they are collinear with origin.
    # This creates a structure of two nested tetrahedra.
    
    # Octahedron vertices (normalized):
    o1 = np.array([1, 0, 0])
    o2 = np.array([-1, 0, 0])
    o3 = np.array([0, 1, 0])
    o4 = np.array([0, -1, 0])
    o5 = np.array([0, 0, 1])
    o6 = np.array([0, 0, -1])
    
    base_vectors = np.vstack([t1, t2, t3, t4, t1, t2, t3, t4, o1, o2, o3, o4, o5, o6])
    
    def _expand_params(params):
        """
        params: [r1, r2, r3]
        r1: radius for first tetrahedron orbit (4 pts)
        r2: radius for second tetrahedron orbit (4 pts)
        r3: radius for octahedron orbit (6 pts)
        """
        r1, r2, r3 = params
        pts = np.zeros((14, 3))
        pts[0:4] = base_vectors[0:4] * r1
        pts[4:8] = base_vectors[4:8] * r2
        pts[8:14] = base_vectors[8:14] * r3
        return pts
    
    # Objective for symmetry-reduced phase
    def _obj_sym(params):
        pts = _expand_params(params)
        # We want to maximize smooth score, so minimize negative
        return -_smooth_score(pts, beta=50.0)
    
    # Initial guesses for radii
    # Try a few different starting points
    best_params = None
    best_score_sym = -np.inf
    
    initial_guesses = [
        [1.0, 1.0, 1.0],
        [1.0, 0.5, 1.0],
        [1.0, 1.5, 0.8],
        [0.8, 1.2, 1.0],
        [1.0, 1.0, 0.5]
    ]
    
    for guess in initial_guesses:
        try:
            res = minimize(_obj_sym, guess, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                pts = _expand_params(res.x)
                s = _score(pts)
                if s > best_score_sym:
                    best_score_sym = s
                    best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to a standard icosahedron subset if symmetry search fails
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        # Add 2 more points, e.g., north and south poles
        pts = np.vstack([ico, np.array([[0, 0, 1], [0, 0, -1]])])
        best_score_sym = _score(pts)
        best_points = pts
    else:
        best_points = _expand_params(best_params)
        best_score_sym = _score(best_points)
    
    # --- Phase 2: Full 42-dim Continuous Optimization (Polish) ---
    # Use the symmetric result as a seed for a general optimization to break symmetry if beneficial.
    
    def _obj_full(x):
        pts = x.reshape(n, d)
        return -_smooth_score(pts, beta=100.0)
    
    x0 = best_points.flatten()
    
    # Run SLSQP on the full 42 dimensions
    # This is computationally expensive, so we limit iterations and use a tight tolerance
    try:
        res_full = minimize(_obj_full, x0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-10})
        if res_full.success:
            pts_full = res_full.x.reshape(n, d)
            s_full = _score(pts_full)
            if s_full > best_score_sym:
                best_points = pts_full
                best_score_sym = s_full
    except Exception:
        pass
    
    # --- Phase 3: High-Precision Polish with Nelder-Mead ---
    # Sometimes gradient methods get stuck in flat regions. A few steps of Nelder-Mead can help.
    if time.time() < deadline:
        def _obj_nm(x):
            pts = x.reshape(n, d)
            return -_score(pts) # Use exact score for final polish, but it's non-smooth.
            # To make it smoother for NM, we could stick to smooth score, but exact is better for final value.
            # However, NM on non-smooth can be erratic. Let's use smooth score with high beta.
        
        def _obj_nm_smooth(x):
            pts = x.reshape(n, d)
            return -_smooth_score(pts, beta=500.0)
            
        try:
            res_nm = minimize(_obj_nm_smooth, best_points.flatten(), method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            pts_nm = res_nm.x.reshape(n, d)
            s_nm = _score(pts_nm)
            if s_nm > best_score_sym:
                best_points = pts_nm
                best_score_sym = s_nm
        except Exception:
            pass

    # Final check and return
    final_score = _score(best_points)
    if not np.isfinite(final_score) or final_score < 0:
        # Fallback to a known good configuration
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0, 0, 1], [0, 0, -1]])])
        
    return best_points

# EVOLVE-BLOCK-END
