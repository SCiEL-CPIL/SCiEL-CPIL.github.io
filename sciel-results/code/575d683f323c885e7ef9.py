"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # seconds wall-clock limit; keep safety margin

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Pre-computed indices for speed if possible, but n=13 is small
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def _hull_area(points):
    """Compute the convex hull area of the point set."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        # If hull fails (e.g., collinear points), return a small value to penalize
        return 1e-6

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return _min_triangle_area(points) / hull_area

def _objective(x):
    """Objective function for minimization: negative score."""
    pts = x.reshape(-1, 2)
    # Simple penalty if points collapse
    if np.any(np.diff(np.sort(pts[:, 0])) < 1e-10) or np.any(np.diff(np.sort(pts[:, 1])) < 1e-10):
         pass # ConvexHull will handle degenerate cases mostly
    s = _score(pts)
    return -s

def _regular_polygon(n, center=(0, 0), radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def construct():
    start_time = time.time()
    
    def _time_remaining():
        return DEADLINE - (time.time() - start_time)

    # 1. Generate diverse initial configurations
    configs = []
    
    # Config A: Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))
    
    # Config B: Regular 12-gon + Center
    pts_12 = _regular_polygon(12, radius=1.0)
    configs.append(np.vstack([pts_12, np.array([0.0, 0.0])]))
    
    # Config C: Perturbed 12-gon + Center
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        pts_pert = pts_12 + rng_local.randn(12, 2) * 0.1
        center = rng_local.randn(2) * 0.1
        configs.append(np.vstack([pts_pert, center]))
        
    # Config D: "Grid-like" structure to break rotational symmetry
    # 3x3 grid scaled, plus 4 points on the perimeter
    grid = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1,  0], [0,  0], [1,  0],
        [-1,  1], [0,  1], [1,  1]
    ])
    # Scale to unit area roughly (area of hull for -1..1 is 4, so scale by 0.5)
    grid = grid * 0.5
    # Add 4 more points on the edges/corners slightly outside or inside to make 13
    # Actually, let's just use a 13-point random cluster near the unit circle
    for seed in [10, 11]:
        rng_rand = np.random.RandomState(seed)
        angles = np.sort(rng_rand.uniform(0, 2*np.pi, 13))
        radii = rng_rand.uniform(0.8, 1.0, 13)
        pts_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts_rand)

    best_points = None
    best_score = -np.inf

    # Evaluate initial configs
    for cfg in configs:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = cfg.copy()
            
    # 2. Optimization Phase
    # Use Basin Hopping to escape local minima
    # This is superior to single Nelder-Mead runs for this problem
    
    # We will run basinhopping on the best initial config and a few others
    # to ensure we don't miss a better basin.
    
    def _local_minimizer(x):
        """Wrapper for Nelder-Mead to be used inside basinhopping."""
        result = minimize(_objective, x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        return result

    # We define a step function for basinhopping
    def _step(x):
        # Random perturbation
        return x + np.random.randn(*x.shape) * 0.1

    # Run basinhopping from the best initial config
    if _time_remaining() > 20.0:
        try:
            x0 = best_points.flatten()
            bh_result = basinhopping(_objective, x0, niter=20, 
                                     stepsize=0.5, 
                                     minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500}},
                                     acceptance_temperature=1.0,
                                     disp=False)
            bh_pts = bh_result.x.reshape(-1, 2)
            bh_score = _score(bh_pts)
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_pts.copy()
        except Exception:
            pass

    # Run basinhopping from the 12-gon+center config specifically, as it's a strong candidate
    if _time_remaining() > 20.0:
        try:
            x0 = configs[1].flatten() # 12-gon + center
            bh_result = basinhopping(_objective, x0, niter=20, 
                                     stepsize=0.5, 
                                     minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500}},
                                     acceptance_temperature=1.0,
                                     disp=False)
            bh_pts = bh_result.x.reshape(-1, 2)
            bh_score = _score(bh_pts)
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_pts.copy()
        except Exception:
            pass

    # 3. Final Polish
    # Run a high-precision Nelder-Mead on the global best
    if _time_remaining() > 5.0:
        try:
            x0 = best_points.flatten()
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # 4. Second Polish with Powell if time allows
    if _time_remaining() > 5.0:
        try:
            x0 = best_points.flatten()
            result = minimize(_objective, x0, method='Powell',
                             options={'maxiter': 1000, 'ftol': 1e-12})
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    return best_points
