"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _symmetric_points(params):
    """
    Construct 11 points with specific symmetry.
    Params: [a1, b1, a2, b2]
    P0..P6 are fixed symmetric points.
    P7, P8 are mirror images at y=b1, x=0.5+-a1.
    P9, P10 are mirror images at y=b2, x=0.5+-a2.
    """
    points = np.zeros((N, 2))
    # Fixed points
    points[0] = [0.0, 0.0]      # Corner A
    points[1] = [1.0, 0.0]      # Corner B
    points[2] = [0.5, H]        # Corner C
    points[3] = [0.5, 0.0]      # Mid AB
    points[4] = [0.25, H/2]     # Mid AC
    points[5] = [0.75, H/2]     # Mid BC
    points[6] = [0.5, H/3]      # Centroid
    
    a1, b1, a2, b2 = params
    
    # Free points with reflectional symmetry about x=0.5
    points[7] = [0.5 - a1, b1]
    points[8] = [0.5 + a1, b1]
    points[9] = [0.5 - a2, b2]
    points[10] = [0.5 + a2, b2]
    
    return points


def _objective_sym(params):
    """Minimize negative min-area for symmetric configuration."""
    points = _symmetric_points(params)
    # Check validity
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    # Define bounds for the 4 symmetric parameters
    # a1, a2 in [0, 0.5]
    # b1, b2 in [0, H]
    bounds = [(0.0, 0.5), (0.0, H), (0.0, 0.5), (0.0, H)]

    # Phase 1: Differential Evolution on the reduced 4D space
    try:
        result = differential_evolution(
            _objective_sym,
            bounds=bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        if time.time() > deadline:
            pass # Just proceed to return best found so far, but DE usually finishes fast
        best_params = result.x
        best_score = -result.fun
    except Exception:
        pass

    # Phase 2: Local polish with Nelder-Mead on the 4D space
    if time.time() < deadline - 10.0 and best_params is not None:
        try:
            res2 = minimize(
                _objective_sym,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if -res2.fun > best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    # Phase 3: Multiple random restarts of local polish if time remains
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(123)
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            params = rng.uniform(0, 0.5, size=2)
            params[1] = rng.uniform(0, H)
            params[3] = rng.uniform(0, H)
            # Ensure b < sqrt(3)*(0.5-a) roughly, though penalty handles it
            try:
                res = minimize(
                    _objective_sym,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if -res.fun > best_score:
                    best_params = res.x
                    best_score = -res2.fun # Correction: best_score = -res.fun
                    best_score = -res.fun
            except Exception:
                pass

    if best_params is not None:
        points = _symmetric_points(best_params)
        # Final safety: ensure all points are strictly inside
        for i in range(N):
            if not _is_inside(points[i]):
                # Nudge inward towards centroid
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        # Absolute fallback: return the fixed symmetric points
        points = _symmetric_points([0.1, 0.2, 0.2, 0.4])
        return points

# EVOLVE-BLOCK-END
