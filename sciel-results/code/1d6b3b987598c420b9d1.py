"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _fibonacci_spiral(n, offset=0.5):
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _thomson_init(n, time_budget=30.0):
    """Thomson-like initialization: repulsive energy minimization."""
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(n)
    # Add small noise
    pts = _normalize(pts + np.random.normal(0, 0.01, (n, 3)))
    
    start = time.time()
    deadline = start + time_budget
    
    # Minimize repulsive energy (Thomson problem)
    def energy(flat):
        p = _normalize(flat.reshape(n, 3))
        # Compute pairwise distances
        dots = p @ p.T
        np.fill_diagonal(dots, 0.0)
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        # Repulsive energy ~ 1/d^2 (or similar)
        # Use a softer version to avoid singularities
        dists_safe = np.maximum(dists, 1e-6)
        E = np.sum(1.0 / (dists_safe ** 2))
        return E
    
    # Gradient-free optimization for Thomson
    try:
        res = minimize(energy, pts.flatten(), method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() < deadline:
            pts = _normalize(res.x.reshape(n, 3))
    except Exception:
        pass
    
    # Quick L-BFGS refinement
    if time.time() < deadline - 5:
        try:
            def energy_lbfgs(flat):
                p = _normalize(flat.reshape(n, 3))
                dots = p @ p.T
                np.fill_diagonal(dots, 0.0)
                dists = np.arccos(np.clip(dots, -1.0, 1.0))
                dists_safe = np.maximum(dists, 1e-6)
                return np.sum(1.0 / (dists_safe ** 2))
            
            res = minimize(energy_lbfgs, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-5, 'maxfun': 100})
            if res.fun < energy_lbfgs(pts.flatten()):
                pts = _normalize(res.x.reshape(n, 3))
        except Exception:
            pass
    
    return pts

def _exact_covering_radius(pts):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of the spherical triangle
            n = np.cross(v1 - v2, v2 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            
            # Ensure c points in the general direction of the vertices
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            
            # Distance from c to any vertex
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        # Fallback to grid-based estimate
        test_points = _fibonacci_spiral(50000)
        dots = test_points @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        max_radius = np.max(min_dists)
        
    return np.degrees(max_radius)

def _covering_radius_soft(flat_pts, test_points, beta=10.0):
    """
    Compute a smooth approximation of the covering radius.
    """
    pts = flat_pts.reshape(N, 3)
    pts = _normalize(pts)
    
    dots = test_points @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    max_d = np.max(min_dists)
    if beta <= 0:
        return max_d
    shifted = min_dists - max_d
    log_sum = max_d + np.log(np.sum(np.exp(shifted)))
    return log_sum / beta

def _optimize_single_start(pts_init, start_time, deadline):
    """Run multi-phase optimization for a single start."""
    best_pts = pts_init.copy()
    
    # Phase 1: Coarse
    if time.time() > deadline - 40: return best_pts
    n_test1 = 10000
    test_pts1 = _fibonacci_spiral(n_test1)
    beta1 = 10.0
    
    def obj1(flat):
        return _covering_radius_soft(flat, test_pts1, beta1)
        
    try:
        res = minimize(obj1, best_pts.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-5, 'maxfun': 200})
        if res.fun < obj1(best_pts.flatten()):
            best_pts = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass

    # Phase 2: Medium
    if time.time() > deadline - 20: return best_pts
    n_test2 = 50000
    test_pts2 = _fibonacci_spiral(n_test2)
    beta2 = 50.0
    
    def obj2(flat):
        return _covering_radius_soft(flat, test_pts2, beta2)
        
    try:
        res = minimize(obj2, best_pts.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 300, 'gtol': 1e-6, 'maxfun': 300})
        if res.fun < obj2(best_pts.flatten()):
            best_pts = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass

    # Phase 3: Fine
    if time.time() > deadline - 5: return best_pts
    n_test3 = 100000
    test_pts3 = _fibonacci_spiral(n_test3)
    beta3 = 200.0
    
    def obj3(flat):
        return _covering_radius_soft(flat, test_pts3, beta3)
        
    try:
        res = minimize(obj3, best_pts.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-7, 'maxfun': 200})
        if res.fun < obj3(best_pts.flatten()):
            best_pts = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass

    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    candidates = []
    
    # Strategy 1: Thomson initialization
    if time.time() < deadline - 60:
        pts_thomson = _thomson_init(N, time_budget=25.0)
        res_pts = _optimize_single_start(pts_thomson, start_time, deadline)
        candidates.append(res_pts)
    
    # Strategy 2: Fibonacci spiral with noise
    if time.time() < deadline - 30:
        offsets = [0.5, 0.2, 0.8]
        for offset in offsets[:2]:  # Try 2 more starts
            if time.time() > deadline - 30:
                break
            pts_init = _fibonacci_spiral(N, offset)
            noise = np.random.normal(0, 0.01, (N, 3))
            pts_init = _normalize(pts_init + noise)
            
            res_pts = _optimize_single_start(pts_init, start_time, deadline)
            candidates.append(res_pts)
    
    if not candidates:
        return _fibonacci_spiral(N)
        
    # Evaluate exact covering radius for all candidates
    best_pts = candidates[0]
    best_radius = _exact_covering_radius(best_pts)
    
    for pts in candidates[1:]:
        rad = _exact_covering_radius(pts)
        if rad < best_radius:
            best_radius = rad
            best_pts = pts
            
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
