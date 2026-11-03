"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 1e-10

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross product|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _soft_min_triangle_area(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    We want to maximize the min area, so we use -soft_min of areas.
    soft_min(a_1,...,a_m) ≈ -(1/beta) * log(sum(exp(-beta*a_i)))
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    areas = np.array(areas)
    # soft min = -(1/beta)*log(sum(exp(-beta*areas)))
    min_area = areas.min()
    # Shift for numerical stability
    shifted = -beta * (areas - min_area)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_area - (1.0/beta) * log_sum
    return soft_min

def _objective(params, beta=50.0):
    """
    Objective function for minimization: negative soft-min triangle area / convex hull area.
    We minimize this to maximize the ratio.
    """
    points = params.reshape(-1, 2)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    soft_min_area = _soft_min_triangle_area(points, beta)
    return -(soft_min_area / hull_area)

def _score(points):
    """Compute the actual score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _fermat_spiral(n):
    """Generate points using Fermat's spiral (Vogel's model)."""
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points

def _circle_initial(n, n_ring=2):
    """Generate points on concentric circles."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    # Outer ring
    remaining = n - 1
    angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
    for a in angles:
        points.append([np.cos(a), np.sin(a)])
    return np.array(points)

def _grid_initial(n):
    """Generate points on a perturbed grid."""
    side = int(np.ceil(np.sqrt(n)))
    grid = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                grid.append([i/(side-1) - 0.5, j/(side-1) - 0.5])
                count += 1
    points = np.array(grid)
    # Add small random perturbation
    rng = np.random.RandomState(42)
    points += rng.normal(0, 0.01, size=points.shape)
    return points

def _optimize_from_start(start_points, time_limit, beta_start=50.0, beta_end=200.0, n_beta_steps=5):
    """
    Optimize from a starting configuration using L-BFGS-B with annealing beta.
    Returns the best points found.
    """
    t0 = time.time()
    best_points = start_points.copy()
    best_score = _score(start_points)
    
    # Anneal beta from low to high for smoothness
    betas = np.linspace(beta_start, beta_end, n_beta_steps)
    
    current_params = start_points.flatten()
    
    for beta in betas:
        if time.time() - t0 > time_limit - 5:
            break
        
        result = minimize(
            _objective,
            current_params,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _objective(current_params, beta):
            current_params = result.x
        
        # Update best
        pts = current_params.reshape(-1, 2)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Final polish with exact min (no smoothing)
    if time.time() - t0 < time_limit - 3:
        # Do a few more iterations with high beta
        for beta in [500.0, 1000.0]:
            if time.time() - t0 > time_limit - 3:
                break
            result = minimize(
                _objective,
                current_params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.success or result.fun < _objective(current_params, beta):
                current_params = result.x
            pts = current_params.reshape(-1, 2)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
    
    return best_points, best_score

def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin before 120s
    
    # Generate multiple diverse starting configurations
    starts = []
    
    # 1. Fermat spiral
    starts.append(_fermat_spiral(N))
    
    # 2. Circle-based
    starts.append(_circle_initial(N))
    
    # 3. Grid-based
    starts.append(_grid_initial(N))
    
    # 4. Random with seed
    rng = np.random.RandomState(123)
    starts.append(rng.random((N, 2)) - 0.5)
    
    # 5. Another Fermat with rotation
    i = np.arange(N)
    r = np.sqrt(i / N)
    theta = i * 2.399963229728653 + np.pi/5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    starts.append(pts)
    
    # 6. Hexagonal-ish arrangement
    starts.append(_fermat_spiral(N) * 1.1 + 0.05 * np.random.RandomState(7).random((N, 2)))
    
    # Optimize from each start
    best_points = starts[0].copy()
    best_score = _score(starts[0])
    
    for idx, start in enumerate(starts):
        remaining = time_limit - (time.time() - t_start)
        if remaining < 10:
            break
        
        # Allocate time per start
        time_per_start = min(remaining / (len(starts) - idx), 18.0)
        
        pts, sc = _optimize_from_start(start, time_per_start)
        
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # Final local polish on the best result
    remaining = time_limit - (time.time() - t_start)
    if remaining > 5:
        # Do a final high-precision optimization
        current_params = best_points.flatten()
        for beta in [2000.0, 5000.0]:
            if time.time() - t_start > time_limit - 2:
                break
            try:
                result = minimize(
                    _objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.success or result.fun < _objective(current_params, beta):
                    current_params = result.x
                pts = current_params.reshape(-1, 2)
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
