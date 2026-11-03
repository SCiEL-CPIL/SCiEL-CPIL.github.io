"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

# EVOLVE-BLOCK-START

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _compute_facet_circumradii(points):
    """
    Compute the circumradius (angular distance from center to facet) for all triangular facets
    of the convex hull of the given points.
    Returns an array of circumradii in radians.
    """
    if len(points) < 3:
        return np.array([])
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([np.pi]) # Fallback if hull fails

    radii = []
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute normal vector to the facet
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            continue
            
        n = n / norm_n
        
        # Ensure normal points outward (away from origin)
        # Since points are on unit sphere, origin is inside hull.
        # Dot product of normal with any point on facet should be positive if normal is outward.
        if np.dot(n, p1) < 0:
            n = -n
            
        # The distance from origin to the plane is h = dot(n, p1)
        # The circumradius of the spherical triangle (cap) is arccos(h)
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        
        # If h is close to 1, the cap is small. If h is close to -1, the cap is large (but shouldn't happen for convex hull of sphere points)
        angle = np.arccos(h)
        radii.append(angle)
        
    if not radii:
        return np.array([np.pi])
        
    return np.array(radii)

def _covering_radius(points):
    """
    Compute the exact covering radius in degrees.
    """
    radii = _compute_facet_circumradii(points)
    if len(radii) == 0:
        return 180.0
    max_rad = np.max(radii)
    return max_rad * 180.0 / np.pi

def _smoothed_covering_radius(params, beta=50.0):
    """
    Smoothed surrogate for the covering radius.
    Uses log-sum-exp on the squared circumradii to approximate the maximum.
    This is differentiable and suitable for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    radii = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return np.pi
        
    # We want to minimize the max radius.
    # log-sum-exp(beta * radii) / beta approximates max(radii)
    # Using squared radii might help with scaling? Let's stick to linear radii first.
    scaled = beta * radii
    max_val = np.max(scaled)
    # Numerical stability
    logsumexp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    smooth_max = logsumexp / beta
    
    return smooth_max

def _generate_initial_points(method='fibonacci', seed=42):
    """Generate initial points on the sphere."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    elif method == 'random':
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
    elif method == 'icosahedron':
        # Icosahedron vertices plus perturbations
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        verts = _normalize(verts)
        # Add 38 more points randomly near the icosahedron vertices? 
        # Or just use random for the rest? 
        # Let's stick to random for simplicity in initial gen, or expand icosahedron.
        # For now, if method is icosahedron, just return icosahedron + random fill?
        # Actually, let's just use random or fibonacci as they are robust starts.
        pts = verts
        rng = np.random.RandomState(seed)
        while len(pts) < N:
            new_pt = rng.randn(3)
            new_pt = _normalize(new_pt.reshape(1,3))[0]
            pts = np.vstack([pts, new_pt])
        pts = pts[:N]
    else:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        
    return _normalize(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s buffer for 150s limit
    
    best_points = None
    best_cost = np.inf
    
    # Strategy:
    # 1. Try several initializations (Fibonacci, Random, etc.)
    # 2. For each, run a local optimization using smoothed objective (L-BFGS-B)
    # 3. Polish the result with Nelder-Mead on the true objective
    # 4. Keep the best result found within the time limit.
    
    initializations = [
        ('fibonacci', 42),
        ('random', 42),
        ('random', 123),
        ('random', 777),
        ('icosahedron', 42),
        ('fibonacci', 123) # Perturbed fibonacci? No, just different seed for random part if any.
    ]
    
    # We can also try a simple simulated annealing step to improve initializations
    # but L-BFGS on smoothed objective is usually very effective.
    
    for method, seed in initializations:
        if time.time() > deadline - 20: # Need at least 20s for optimization
            break
            
        init_pts = _generate_initial_points(method, seed)
        
        # Optional: Small SA step to break symmetry/improve start
        # Let's skip SA for now to save time, L-BFGS is strong.
        
        # Phase 1: L-BFGS-B on smoothed objective
        # We need to flatten the points
        x0 = init_pts.flatten()
        
        def objective_smoothed(params):
            return _smoothed_covering_radius(params, beta=50.0)
        
        try:
            res = minimize(objective_smoothed, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            pts_smooth = _normalize(res.x.reshape(N, 3))
        except Exception:
            pts_smooth = init_pts
            
        cost_smooth = _covering_radius(pts_smooth)
        
        # Phase 2: Nelder-Mead on true objective for polishing
        # This fixes the discrepancy between smoothed and true objective
        x1 = pts_smooth.flatten()
        
        def objective_true(params):
            pts = _normalize(params.reshape(N, 3))
            return _covering_radius(pts)
            
        try:
            res2 = minimize(objective_true, x1, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': 2000})
            pts_final = _normalize(res2.x.reshape(N, 3))
        except Exception:
            pts_final = pts_smooth
            
        cost_final = _covering_radius(pts_final)
        
        if cost_final < best_cost:
            best_cost = cost_final
            best_points = pts_final.copy()
            
        # Check time
        if time.time() > deadline - 5:
            break

    # If we still have time, do a final high-precision polish on the best candidate
    if best_points is not None and time.time() < deadline - 5:
        x_best = best_points.flatten()
        def objective_true(params):
            pts = _normalize(params.reshape(N, 3))
            return _covering_radius(pts)
            
        try:
            # Tighter tolerance
            res3 = minimize(objective_true, x_best, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 1000})
            pts_polished = _normalize(res3.x.reshape(N, 3))
            cost_polished = _covering_radius(pts_polished)
            if cost_polished < best_cost:
                best_cost = cost_polished
                best_points = pts_polished.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = _generate_initial_points('fibonacci', 42)
        
    return _normalize(best_points)

# EVOLVE-BLOCK-END
