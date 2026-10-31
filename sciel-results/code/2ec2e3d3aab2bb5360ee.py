"""Heilbronn-type problem: place 13 points in unit-area convex region to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time
import itertools

N = 13

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    # Vectorized: compute all triangle areas at once using cross product
    # For each triple (i, j, k), area = 0.5 * |cross(pj-pi, pk-pi)|
    for i in range(n - 2):
        pi = points[i]
        # Points from i+1 to end
        rest = points[i+1:]
        # For each pair (j, k) in rest
        m = len(rest)
        for j in range(m - 1):
            pj = rest[j]
            # All k > j
            pk_arr = rest[j+1:]
            # cross(pj - pi, pk - pi) for all pk
            v1 = pj - pi
            v2 = pk_arr - pi
            cross_vals = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(x):
    """Objective for minimization (we want to maximize score, so negate)."""
    points = x.reshape(N, 2)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Strategy: Use differential evolution for global search, then local refinement.
    # The problem is to maximize min_triangle_area / hull_area.
    # Since the score is scale-invariant (both areas scale with the square of scaling),
    # we can fix the hull area to 1 by normalizing, or just work in raw coordinates.
    # Actually, the score is ratio, so it's invariant under scaling. 
    # Let's work in a bounded box and let the optimizer figure out the geometry.
    
    # Bounds: points in [0, 10] x [0, 10] box
    bounds = [(0, 10)] * (2 * N)
    
    # Run differential evolution with multiple seeds/strategies
    rng = np.random.RandomState(42)
    
    # First, try a few quick DE runs with different populations
    de_kwargs = {
        'bounds': bounds,
        'maxiter': 200,
        'popsize': 15,
        'tol': 1e-6,
        'seed': 42,
        'polish': False,
        'init': 'latinhypercube',
        'mutation': (0.5, 1.0),
        'recombination': 0.7,
        'updating': 'immediate',
        'strategy': 'best1bin',
    }
    
    # Run DE
    try:
        result = differential_evolution(_objective, bounds, **de_kwargs)
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            points = result.x.reshape(N, 2)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    except Exception:
        pass
    
    # Try a second DE run with different parameters if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            de_kwargs2 = {
                'bounds': bounds,
                'maxiter': 150,
                'popsize': 20,
                'tol': 1e-5,
                'seed': 123,
                'polish': False,
                'init': 'sobol',
                'mutation': (0.8, 1.2),
                'recombination': 0.8,
                'updating': 'immediate',
                'strategy': 'best1bin',
            }
            result2 = differential_evolution(_objective, bounds, **de_kwargs2)
            points2 = result2.x.reshape(N, 2)
            s2 = _score(points2)
            if s2 > best_score:
                best_score = s2
                best_points = points2.copy()
        except Exception:
            pass
    
    # Local refinement: use Nelder-Mead or Powell on the best solution
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline - 5:
        try:
            from scipy.optimize import minimize
            # Use Powell method for local refinement
            res_local = minimize(
                _objective, 
                best_points.flatten(), 
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            points_local = res_local.x.reshape(N, 2)
            s_local = _score(points_local)
            if s_local > best_score:
                best_score = s_local
                best_points = points_local.copy()
        except Exception:
            pass
    
    # If we still have time, try another local refinement with different starting points
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline - 10:
        try:
            # Perturb the best solution and do local optimization
            rng_local = np.random.RandomState(7)
            for _ in range(3):
                if time.time() - start_time > deadline - 5:
                    break
                perturbation = rng_local.normal(0, 0.5, size=(N, 2))
                x0 = (best_points + perturbation).flatten()
                res_pert = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                points_pert = res_pert.x.reshape(N, 2)
                s_pert = _score(points_pert)
                if s_pert > best_score:
                    best_score = s_pert
                    best_points = points_pert.copy()
        except Exception:
            pass
    
    # Final check: if no valid solution found, return a reasonable default
    if best_points is None:
        # Return points on a regular grid-like pattern
        best_points = np.array([[i % 4, i // 4] for i in range(N)], dtype=float)
    
    return best_points
