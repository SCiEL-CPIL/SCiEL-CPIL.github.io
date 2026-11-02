"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

def _all_triangle_areas(pts):
    """Compute all triangle areas among triples of points, vectorized."""
    n = len(pts)
    if n < 3:
        return np.array([])
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas.append(0.5 * cross)
    
    if len(areas) == 0:
        return np.array([])
    return np.concatenate(areas)

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _neg_score_params(params, generator_func):
    """Negative score for minimization, using a generator function."""
    try:
        pts = generator_func(params)
        if pts is None or len(pts) != N:
            return 0.0
        score = _score_points(pts)
        return -score
    except Exception:
        return 0.0

def _gen_1_6_6(params):
    """1 center, 6 inner, 6 outer."""
    r1, r2, phi = params
    pts = [[0.0, 0.0]]
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for a in angles_inner:
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + phi
    for a in angles_outer:
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts)

def _gen_1_5_7(params):
    """1 center, 5 inner, 7 outer."""
    r1, r2, phi = params
    pts = [[0.0, 0.0]]
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    for a in angles_inner:
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + phi
    for a in angles_outer:
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts)

def _gen_1_4_8(params):
    """1 center, 4 inner, 8 outer."""
    r1, r2, phi = params
    pts = [[0.0, 0.0]]
    angles_inner = np.linspace(0, 2*np.pi, 4, endpoint=False)
    for a in angles_inner:
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + phi
    for a in angles_outer:
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts)

def _gen_1_3_9(params):
    """1 center, 3 inner, 9 outer."""
    r1, r2, phi = params
    pts = [[0.0, 0.0]]
    angles_inner = np.linspace(0, 2*np.pi, 3, endpoint=False)
    for a in angles_inner:
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    angles_outer = np.linspace(0, 2*np.pi, 9, endpoint=False) + phi
    for a in angles_outer:
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts)

def _gen_1_12(params):
    """1 center, 12 outer."""
    r = params[0]
    pts = [[0.0, 0.0]]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for a in angles:
        pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts)

def _gen_13(params):
    """13 points on a ring."""
    r = params[0]
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = []
    for a in angles:
        pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts)

def _gen_2_5_6(params):
    """2 centers? No, let's stick to 1 center."""
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_pts = None
    best_params = None
    best_gen = None
    
    # List of (generator, bounds, initial_guess)
    # Radii should be positive. Angles 0 to pi.
    # We want the hull to be reasonably large, so r2 should be > r1.
    
    configs = [
        {
            'name': '1_6_6',
            'gen': _gen_1_6_6,
            'bounds': [(0.1, 1.0), (0.5, 2.0), (0.0, np.pi)],
            'seed': np.array([0.5, 1.0, 0.0])
        },
        {
            'name': '1_5_7',
            'gen': _gen_1_5_7,
            'bounds': [(0.1, 1.0), (0.5, 2.0), (0.0, np.pi)],
            'seed': np.array([0.5, 1.0, 0.0])
        },
        {
            'name': '1_4_8',
            'gen': _gen_1_4_8,
            'bounds': [(0.1, 1.0), (0.5, 2.0), (0.0, np.pi)],
            'seed': np.array([0.5, 1.0, 0.0])
        },
        {
            'name': '1_3_9',
            'gen': _gen_1_3_9,
            'bounds': [(0.1, 1.0), (0.5, 2.0), (0.0, np.pi)],
            'seed': np.array([0.5, 1.0, 0.0])
        },
        {
            'name': '1_12',
            'gen': _gen_1_12,
            'bounds': [(0.5, 2.0)],
            'seed': np.array([1.0])
        },
        {
            'name': '13',
            'gen': _gen_13,
            'bounds': [(0.5, 2.0)],
            'seed': np.array([1.0])
        }
    ]
    
    # Run Differential Evolution for each config
    for cfg in configs:
        if time.time() > deadline - 10:
            break
            
        try:
            result = differential_evolution(
                lambda x: _neg_score_params(x, cfg['gen']),
                bounds=cfg['bounds'],
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-4,
                polish=False
            )
            
            if result.fun is not None:
                candidate_pts = cfg['gen'](result.x)
                if candidate_pts is not None and len(candidate_pts) == N:
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        best_params = result.x.copy()
                        best_gen = cfg['gen']
                        
        except Exception:
            pass
    
    # Local refinement on the best found symmetric configuration
    # We can try to optimize the parameters again with a tighter DE or Nelder-Mead on params
    if best_params is not None and best_gen is not None:
        if time.time() < deadline - 5:
            # Refine parameters with Nelder-Mead
            try:
                res = minimize(
                    lambda x: _neg_score_params(x, best_gen),
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                candidate_pts = best_gen(res.x)
                if candidate_pts is not None and len(candidate_pts) == N:
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
            except Exception:
                pass

    # Final check: try to break symmetry slightly from the best point set
    # This is risky and might not help, but if we have time...
    if best_pts is not None and time.time() < deadline - 2:
        try:
            # Nelder-Mead on full 26D
            res = minimize(
                lambda x: -_score_points(x.reshape(-1, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass

    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

# EVOLVE-BLOCK-END
