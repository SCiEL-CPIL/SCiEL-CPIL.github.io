"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

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

def _smooth_min_objective(x, tau=20.0):
    """
    Objective for minimization: -smooth_min(area_i / hull_area).
    Uses log-sum-exp trick for numerical stability.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
        
    # Normalize areas by hull area to make objective scale-invariant
    ratios = areas / hull_area
    
    # Smooth minimum: min(x) ~ -1/tau * log(sum(exp(-tau * x)))
    # We want to maximize min(ratios), so minimize -min(ratios)
    # Using log-sum-exp for stability:
    # min(ratios) approx min_r - (1/tau) * log(sum(exp(-tau * (ratios - min_r))))
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    # Clip to prevent overflow in exp
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_min_val = min_r - (1.0 / tau) * log_sum
    
    return -smooth_min_val

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _nelder_mead_polish(pts, max_iter=200, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Define bounds for DE: points in [-1, 1] x [-1, 1]
    # This covers a square of area 4. The hull area will vary.
    # Since the score is normalized by hull area, the absolute scale doesn't matter,
    # but we need to ensure points don't collapse.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Phase 1: Differential Evolution for Global Optimization
    # DE is robust to local minima and handles the non-smoothness of the hull area
    # better than gradient-based methods when combined with the smoothed objective.
    try:
        de_result = differential_evolution(
            _smooth_min_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.9,
            polish=False, # We will polish separately with NM
            updating='deferred',
            workers=1
        )
        
        if de_result.success or de_result.fun < 1e5:
            pts = de_result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Phase 2: Nelder-Mead Polish on the best DE result
    if time.time() < deadline - 10:
        polished = _nelder_mead_polish(best_pts, max_iter=300)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 3: Local Refinement with L-BFGS-B using Smoothed Objective
    # Sometimes DE gets stuck near a local optimum where gradients can help.
    if time.time() < deadline - 5:
        try:
            x0 = best_pts.flatten()
            lbfgs_result = minimize(
                _smooth_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts = lbfgs_result.x.reshape(-1, 2)
            
            # Final NM polish on the L-BFGS result
            pts = _nelder_mead_polish(pts, max_iter=100)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception as e:
            pass

    return best_pts
# EVOLVE-BLOCK-END
