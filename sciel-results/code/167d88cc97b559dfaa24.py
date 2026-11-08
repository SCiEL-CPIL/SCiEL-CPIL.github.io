"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by pairs of points."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    return np.array(areas)

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = np.min(_all_triangle_areas(pts))
    return min_area / hull_a

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _multi_beta_soft_obj(pts_flat, betas):
    """
    Objective: -soft_min_triangle_area / hull_area
    Uses multi-scale log-sum-exp to approximate min area smoothly.
    """
    pts = pts_flat.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 1e6
    
    areas = _all_triangle_areas(pts)
    
    # Multi-scale soft min
    # We want to maximize the min area. 
    # soft_min(x, b) = -1/b * log(sum(exp(-b*x)))
    # We use a combination of betas to balance smoothness and approximation of min.
    # Larger b approximates min better but is less smooth.
    
    soft_min_val = 0.0
    for b in betas:
        # Numerical stability: subtract max
        shifted = -b * areas
        shifted -= np.max(shifted)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val -= (1.0/b) * log_sum
        
    # Average the soft mins from different scales
    soft_min_avg = soft_min_val / len(betas)
    
    return -soft_min_avg / hull_a

def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _initial_mixed(n):
    """Mix of boundary and interior points."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)

def _initial_hexagonal(n):
    """Hexagonal lattice-like initial points."""
    pts = []
    # Create a grid of points and take the first n
    # Hexagonal packing
    dx = 1.0
    dy = np.sqrt(3)/2
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * dx + (j % 2) * dx/2
            y = j * dy
            pts.append([x, y])
    pts = np.array(pts[:n])
    return _normalize_unit_area(pts)

def _initial_cosine_mod(n):
    """Perturbed regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    radii = 1.0 + 0.2 * np.cos(3 * angles)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Betas for multi-scale soft min
    # Small beta: smooth, captures "average" small areas
    # Large beta: sharp, approximates true min
    betas = [100, 500, 2000]
    
    best_pts = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    initializers = [
        _initial_regular_polygon,
        _initial_mixed,
        _initial_hexagonal,
        _initial_cosine_mod
    ]
    
    # Add some random initializations
    for _ in range(4):
        def random_init(n):
            pts = rng.uniform(-1, 1, size=(n, 2))
            return _normalize_unit_area(pts)
        initializers.append(random_init)
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            pts = init_func(N)
        except Exception:
            continue
            
        x0 = pts.flatten()
        
        # Run L-BFGS-B optimization
        # We maximize min_area / hull_area, so we minimize -min_area / hull_area
        try:
            res = minimize(
                _multi_beta_soft_obj,
                x0,
                args=(betas,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                opt_pts = res.x.reshape(-1, 2)
                # Normalize to unit area for final score
                opt_pts = _normalize_unit_area(opt_pts)
                score = _score(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts
        except Exception:
            continue
    
    # If no optimization succeeded, fallback to best initial
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
        best_score = _score(best_pts)
        
    # Final polish with a very high beta to sharpen the min approximation
    if time.time() - start_time < DEADLINE - 2 and best_pts is not None:
        sharp_betas = [5000, 10000]
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _multi_beta_soft_obj,
                x0,
                args=(sharp_betas,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                opt_pts = res.x.reshape(-1, 2)
                opt_pts = _normalize_unit_area(opt_pts)
                score = _score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
