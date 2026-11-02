"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 145.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    if min_ta <= 0:
        return 0.0
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _full_objective(x):
    """Objective function for full 26D optimization."""
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        return 0.0
    pts = x.reshape(-1, 2)
    return -_score(pts)


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    # Start with regular 13-gon as baseline
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Phase 1: Differential Evolution in full 26D space
    # This is a structural change: global search without symmetry assumptions
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 30:
        # Bounds: each coordinate in [-2, 2]
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        def de_objective(x):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                return 0.0
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.fun < -best_score:
                candidate = de_result.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement from best DE result
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 10:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _full_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < -best_score:
                candidate = res.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass
    
    # Phase 3: Try multiple random restarts with Nelder-Mead if time remains
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 5:
        rng = np.random.RandomState(123)
        for restart in range(3):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            
            # Random initialization near unit circle
            x0 = rng.uniform(-1.5, 1.5, size=2 * N)
            try:
                res = minimize(
                    _full_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < -best_score:
                    candidate = res.x.reshape(-1, 2)
                    cand_score = _score(candidate)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate
            except Exception:
                pass
    
    # Phase 4: Final polish
    if time.time() - START_TIME < DEADLINE - SAFETY_MARGIN - 3:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _full_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < -best_score:
                candidate = res.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass
    
    # Fallback check
    final_score = _score(best_pts)
    if final_score <= 0:
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
