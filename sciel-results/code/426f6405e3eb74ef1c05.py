"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        if len(points) < 3:
            return 0.0
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon(n):
    """Generate vertices of a regular n-gon on the unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def _params_to_points(params):
    """
    Convert parameters to points.
    params[0]: global rotation
    params[1:]: radial modulation (1 + modulation)
    """
    rot = params[0]
    mods = params[1:]
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + rot
    
    radii = 1.0 + mods
    # Ensure radii are positive to avoid degenerate points at origin causing issues
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))


def _objective(params):
    """
    Objective function for optimization: minimize negative score.
    """
    try:
        points = _params_to_points(params)
        hull_area = _convex_hull_area(points)
        if hull_area < 1e-10:
            return 1e10
        min_area = _min_triangle_area(points)
        return -min_area / hull_area
    except Exception:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Initial best: Regular polygon
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Parameters: [rotation, mod_1, ..., mod_12] (13 params total)
    # We fix mod_13 = 0 implicitly by only optimizing 12 mods + 1 rot? 
    # Actually, let's optimize all 13 mods + 1 rot = 14 params.
    # But due to rotational symmetry, the relative angles matter.
    # Let's use 13 radial mods and 1 rotation.
    
    # Bounds for Differential Evolution
    # Rotation: [0, 2*pi]
    # Modulation: [-0.5, 0.5] (radii from 0.5 to 1.5)
    bounds = [(0, 2 * np.pi)] + [(-0.5, 0.5)] * N
    
    # Phase 1: Differential Evolution for global search
    # DE is slow, so use small population and maxiter
    try:
        if time.time() < deadline - 30:
            res_de = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                polish=False, # We will polish later
                init='sobol'
            )
            if res_de.fun < -best_score:
                best_params = res_de.x
                best_points = _params_to_points(best_params)
                best_score = -res_de.fun
    except Exception:
        pass

    # Phase 2: Local refinement using Nelder-Mead on the best params from DE
    # Also try refining the regular polygon params (all mods 0)
    
    candidate_params_list = []
    
    # 1. Best from DE
    if 'best_params' in locals():
        candidate_params_list.append(best_params)
    
    # 2. Regular polygon (rotation 0, mods 0)
    candidate_params_list.append(np.array([0.0] + [0.0]*N))
    
    # 3. Random perturbations of regular polygon
    rng = np.random.default_rng(123)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        rot = rng.uniform(0, 2*np.pi)
        mods = rng.normal(0, 0.1, size=N)
        candidate_params_list.append(np.concatenate(([rot], mods)))

    for params0 in candidate_params_list:
        if time.time() > deadline - 5:
            break
        
        # Nelder-Mead refinement
        try:
            res_nm = minimize(
                _objective, 
                params0, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_params = res_nm.x
                best_points = _params_to_points(best_params)
                best_score = -res_nm.fun
        except Exception:
            continue

    # Final check: Ensure the returned points are valid
    # If score is still very low, fallback to regular polygon
    if best_score < _score(_regular_polygon(N)):
        best_points = _regular_polygon(N)

    return best_points

# EVOLVE-BLOCK-END
