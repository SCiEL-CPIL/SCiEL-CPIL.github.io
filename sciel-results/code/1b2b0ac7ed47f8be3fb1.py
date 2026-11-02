# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0
N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        if m < 2:
            continue
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
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
    if hull_area <= 1e-9:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area

def _neg_score_flat(x):
    """Objective for optimizers: negative score of flattened points."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _soft_neg_score(x, beta):
    """Soft-min objective for gradient-based methods."""
    pts = x.reshape(N, 2)
    n = len(pts)
    if n < 3:
        return 0.0
    # Compute all triangle areas
    areas = []
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        if m < 2:
            continue
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            areas.extend(0.5 * np.abs(crosses))
    
    if not areas:
        return 0.0
    
    areas = np.array(areas)
    # Soft min
    # To avoid overflow, shift by max
    max_a = np.max(areas)
    soft_min_val = max_a - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - max_a))))
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    
    return -(soft_min_val / hull_area)

def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    best_pts = None
    best_score = 0.0
    
    # 1. Differential Evolution for Global Search
    # Bounds: points in [-2, 2] x [-2, 2]
    bounds = [(-2, 2)] * (2 * N)
    
    # Run DE
    try:
        # Maximize score -> Minimize -score
        # DE is stochastic, so we might need to run it for a while
        # popsize=15, tol=1e-8, maxiter=100
        result_de = differential_evolution(
            _neg_score_flat,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            polish=False # We will polish manually
        )
        if result_de.success or result_de.fun < 0:
            cand_pts = result_de.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
    except Exception:
        pass

    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        if best_pts is None:
            # Fallback to regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return best_pts

    # 2. Local Refinement with Powell (Derivative-free, robust)
    if best_pts is not None:
        try:
            result_powell = minimize(
                _neg_score_flat,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            cand_pts = result_powell.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass

    # 3. Local Refinement with L-BFGS-B (Gradient-based, using soft-min)
    # We try a few betas
    if best_pts is not None:
        for beta in [10, 50, 100, 200]:
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            try:
                def obj(x):
                    return _soft_neg_score(x, beta)
                
                result_lbgfs = minimize(
                    obj,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                cand_pts = result_lbgfs.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
            except Exception:
                pass

    # 4. Final Polish with Powell again
    if best_pts is not None and time.time() - START_TIME < DEADLINE - SAFETY_MARGIN:
        try:
            result_powell2 = minimize(
                _neg_score_flat,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            cand_pts = result_powell2.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        except Exception:
            pass

    # Fallback if all failed
    if best_pts is None or _score(best_pts) <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
# EVOLVE-BLOCK-END
