# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 575.0  # Safety margin before 630s limit

# Precompute triangle indices for fast scoring
_N = 13
_TRIANGLE_INDICES = np.array(list(itertools.combinations(range(_N), 3)))

def _min_triangle_area_fast(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Uses precomputed indices for speed.
    """
    p1 = points[_TRIANGLE_INDICES[:, 0]]
    p2 = points[_TRIANGLE_INDICES[:, 1]]
    p3 = points[_TRIANGLE_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area_fast(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective(x):
    """
    Objective function to minimize (negative of the score).
    """
    points = x.reshape(_N, 2)
    return -_score(points)

def _de_callback(xk, convergence):
    """
    Callback for differential evolution to check time budget.
    """
    if time.time() > _DEADLINE - 120:
        raise RuntimeError("Time budget exceeded")

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = _N
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Differential Evolution for global exploration
    # Search in a bounded box; normalize later
    bounds = [(-1, 1)] * (2 * n)
    
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=25,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,  # We'll do our own polish
            callback=_de_callback,
            updating='deferred',
            init='sobol'
        )
        
        if de_result.x is not None:
            de_points = de_result.x.reshape(n, 2)
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
                
    except Exception as e:
        pass
    
    # Phase 2: Local refinement using Powell from DE result
    if best_points is not None and time.time() < _DEADLINE - 60:
        try:
            res_powell = minimize(
                _objective,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xtol': 1e-12,
                    'ftol': 1e-14,
                    'niter': 200
                }
            )
            pw_score = -res_powell.fun
            if pw_score > best_score:
                best_score = pw_score
                best_points = res_powell.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 3: Additional local refinement with Nelder-Mead
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            res_nm = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            nm_score = -res_nm.fun
            if nm_score > best_score:
                best_score = nm_score
                best_points = res_nm.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 4: Try a few structured starts with Powell as backup
    if time.time() < _DEADLINE - 60:
        for trial in range(5):
            if time.time() > _DEADLINE - 30:
                break
            
            # Structured initial guess: regular polygon + interior
            n_outer = 7
            n_inner = 6
            angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
            outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
            inner_r = 0.3 + rng.random() * 0.3
            inner = inner_r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            pts = np.vstack([outer, inner]) + rng.random((n, 2)) * 0.05
            
            try:
                res = minimize(
                    _objective,
                    pts.flatten(),
                    method='Powell',
                    options={
                        'maxiter': 3000,
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                        'niter': 150
                    }
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_points = res.x.reshape(n, 2)
            except Exception:
                continue
    
    # Fallback
    if best_points is None:
        best_points = rng.random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
