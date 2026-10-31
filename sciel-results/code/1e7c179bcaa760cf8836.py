"""Optimized program for Heilbronn convex regions - 13 points maximizing min triangle area / hull area.
Uses Differential Evolution with smooth-min objective for global search, then local refinement."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 13
MAX_TIME = 145  # Safety margin before 150s limit

def compute_all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = points.shape[0]
    # All pairs of points
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only consider i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    
    v1 = points[jj] - points[ii]
    v2 = points[kk] - points[ii]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def compute_min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    areas = compute_all_triangle_areas(points)
    return np.min(areas)

def compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = compute_min_triangle_area(points)
    return min_area / hull_area

def soft_min_score(params, beta=50.0):
    """Smooth approximation of negative min-triangle-area / hull-area.
    Uses log-sum-exp for smooth min."""
    points = params.reshape(N, 2)
    
    # Center and scale to avoid degenerate configurations
    center = points.mean(axis=0)
    points_c = points - center
    scale = np.max(np.abs(points_c))
    if scale < 1e-10:
        return 1e10
    
    hull_area = compute_hull_area(points_c)
    if hull_area < 1e-10:
        return 1e10
    
    areas = compute_all_triangle_areas(points_c)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # This approximates min(areas) as beta -> infinity
    beta_areas = -beta * areas
    max_val = np.max(beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(beta_areas - max_val)))
    soft_min = -log_sum / beta
    
    score = soft_min / hull_area
    return -score

def local_refine(points, deadline):
    """Local optimization using Nelder-Mead from a good starting point."""
    start_time = time.time()
    current = points.copy()
    current_score = score_function(current)
    
    best_points = current.copy()
    best_score = current_score
    
    # Nelder-Mead with tighter tolerances
    try:
        result = minimize(
            lambda p: -score_function(p.reshape(N, 2)),
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        new_points = result.x.reshape(N, 2)
        new_score = score_function(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    except Exception:
        pass
    
    # Second pass with even tighter tolerances
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                lambda p: -score_function(p.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            new_points2 = result2.x.reshape(N, 2)
            new_score2 = score_function(new_points2)
            if new_score2 > best_score:
                best_points = new_points2
                best_score = new_score2
        except Exception:
            pass
    
    return best_points, best_score

def generate_seed_points(seed=42):
    """Generate diverse seed points for DE population."""
    rng = np.random.RandomState(seed)
    # Mix of structured and random
    points = []
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    points.append(pts_poly)
    # Grid-like
    pts_grid = rng.rand(N, 2) * 2 - 1
    points.append(pts_grid)
    # Hexagonal
    pts_hex = [[0, 0]]
    for i in range(6):
        angle = i * np.pi / 3
        pts_hex.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts_hex.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    points.append(np.array(pts_hex))
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution for global search
    # Use a bounded box to keep points in a reasonable region
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Reduce population size for speed, use fewer generations
    de_popsize = 15
    de_maxiter = 30
    
    try:
        de_result = differential_evolution(
            soft_min_score,
            bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-8,
            mutation=(0.3, 0.9),
            recombination=0.7,
            seed=42,
            polish=True,
            init='latinhypercube',
            update_interval=1,
            callback=lambda xk, convergence: (time.time() > deadline - 10)
        )
        
        if de_result.success or de_result.fun < 0:
            de_points = de_result.x.reshape(N, 2)
            de_score = score_function(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
    except Exception:
        pass
    
    # Phase 2: Local refinement from DE result and other good starts
    candidates = []
    
    if best_points is not None:
        candidates.append(best_points.copy())
    
    # Add some structured starting points
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    rng = np.random.RandomState(123)
    candidates.append(rng.rand(N, 2) * 2 - 1)
    
    # Hexagonal
    pts_hex = [[0, 0]]
    for i in range(6):
        angle = i * np.pi / 3
        pts_hex.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts_hex.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    candidates.append(np.array(pts_hex))
    
    # Try local refinement from each candidate
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        
        refined, score = local_refine(cand, deadline)
        if score > best_score:
            best_score = score
            best_points = refined
    
    # Phase 3: One more DE run with smaller range around best if time permits
    if best_points is not None and time.time() < deadline - 30:
        try:
            center = best_points.mean(axis=0)
            spread = np.std(best_points, axis=0)
            half_range = np.maximum(0.1, spread.max() * 0.5)
            bounds2 = [(center[i] - half_range, center[i] + half_range) for i in range(2 * N)]
            
            de_result2 = differential_evolution(
                soft_min_score,
                bounds2,
                maxiter=15,
                popsize=10,
                tol=1e-10,
                mutation=(0.2, 0.7),
                recombination=0.8,
                seed=777,
                polish=True,
                init='latinhypercube',
                callback=lambda xk, convergence: (time.time() > deadline - 10)
            )
            
            if de_result2.success or de_result2.fun < 0:
                de_points2 = de_result2.x.reshape(N, 2)
                de_score2 = score_function(de_points2)
                if de_score2 > best_score:
                    best_score = de_score2
                    best_points = de_points2
                
                # Refine
                if time.time() < deadline - 10:
                    refined2, score2 = local_refine(de_points2, deadline)
                    if score2 > best_score:
                        best_score = score2
                        best_points = refined2
        except Exception:
            pass
    
    # Final safety: if no valid points found, return a default
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
