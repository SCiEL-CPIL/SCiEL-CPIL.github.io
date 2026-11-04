"""Rewritten program using Differential Evolution on radial parameters for convex polygon vertices."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_TRIPLETS = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _radial_to_cartesian(radii, angles):
    """Convert radial parameters to Cartesian coordinates."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _negative_score_from_radii(radii, angles):
    """Objective function for Differential Evolution (minimization)."""
    pts = _radial_to_cartesian(radii, angles)
    # Ensure points are in convex position (sorted by angle)
    # Since angles are fixed and sorted, and radii > 0, the polygon is convex 
    # if the angles are strictly increasing and we don't have collinearities causing issues.
    # ConvexHull will handle the area calculation correctly regardless of input order, 
    # but for the "min triangle area" to be meaningful in the context of a convex region,
    # the points should generally be in convex position.
    # With fixed sorted angles and positive radii, the points form a star-shaped polygon.
    # To ensure it's a convex polygon, we rely on the fact that if angles are well-spaced 
    # and radii don't vary too wildly, it remains convex. 
    # However, a "star" polygon with concave vertices would have a smaller hull area 
    # than the triangle areas calculated on the star vertices? 
    # Actually, the problem asks for points in a convex region. The hull of the points 
    # is the convex region. If the points are not in convex position (i.e., some are inside),
    # the hull area is determined by the outer points. The min triangle area might involve 
    # interior points, which is allowed. 
    # So we just calculate the score directly.
    return -_score(pts)

def _nelder_mead_polish(pts, max_iter=500, tol=1e-12):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            lambda x: -_score(x.reshape(-1, 2)),
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Fixed angles for 13 points, roughly evenly spaced
    # Add small random perturbations to break symmetry
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = 0.0
    best_pts = None
    
    # Strategy: Differential Evolution on radii
    # Bounds for radii: [0.5, 1.5] to allow some variation around unit circle
    # We will normalize the final result, so absolute scale doesn't matter, 
    # but relative radii do.
    bounds = [(0.2, 2.0)] * N
    
    # Run DE with different angle perturbations to explore the space
    # DE is stochastic, so multiple runs with different seeds/angles help.
    
    n_de_runs = 5
    for run_idx in range(n_de_runs):
        if time.time() > deadline - 60:
            break
            
        # Perturb angles slightly to avoid symmetric local minima
        rng = np.random.RandomState(42 + run_idx)
        angle_perturb = rng.uniform(-0.05, 0.05, N)
        angles = base_angles + angle_perturb
        # Sort angles to maintain convex ordering
        angles = np.sort(angles)
        
        try:
            # Differential Evolution
            # maxiter is the max number of generations
            # popsize is the population size multiplier
            de_result = differential_evolution(
                _negative_score_from_radii,
                bounds,
                args=(angles,),
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42 + run_idx,
                polish=True  # Use Nelder-Mead to polish the DE result
            )
            
            radii_opt = de_result.x
            pts = _radial_to_cartesian(radii_opt, angles)
            
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue

    # If DE didn't find a good solution or if we have time, do a direct Cartesian polish
    # on the best solution found so far
    if best_pts is not None and time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_pts, max_iter=1000)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()

    # Fallback: If something went wrong, return a regular polygon
    if best_pts is None:
        angles = base_angles
        radii = np.ones(N)
        best_pts = _radial_to_cartesian(radii, angles)
        
    return best_pts
