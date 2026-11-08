"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """Calculate minimum triangle area using vectorized cross products."""
    n = len(points)
    min_area = np.inf
    # Precompute all pair vectors
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            # Vectorized: compute cross with all k > j
            if j + 1 < n:
                rest = points[j+1:]
                v2 = rest - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                if areas.size > 0:
                    local_min = np.min(areas)
                    if local_min < min_area:
                        min_area = local_min
    return min_area

def _convex_hull_area_fast(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the normalized minimum triangle area."""
    hull_area = _convex_hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = _min_triangle_area_fast(points)
    return min_tri_area / hull_area

def _params_to_points(params):
    """Convert angular/radial perturbation params to points.
    params: (2N,) array = [dtheta_0, dr_0, dtheta_1, dr_1, ...]
    Base: regular N-gon on unit circle.
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    dtheta = params[0::2]
    dr = params[1::2]
    angles = base_angles + dtheta
    radii = 1.0 + dr
    # Clamp radii to ensure positive
    radii = np.maximum(radii, 0.3)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _negative_score_from_params(params):
    """Objective for optimizer: negative score from parametrized points."""
    points = _params_to_points(params)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with regular polygon (zero perturbations)
    zero_params = np.zeros(2 * N)
    best_points = _params_to_points(zero_params)
    best_score = _score(best_points)
    
    # Phase 1: Differential Evolution on perturbation parameters
    # Bounds: angles in [-pi/N*0.8, pi/N*0.8], radii in [-0.4, 0.4]
    angle_bound = np.pi / N * 0.8
    bounds = [( -angle_bound, angle_bound), (-0.4, 0.4)] * N
    
    try:
        de_result = differential_evolution(
            _negative_score_from_params,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 0.9),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        if de_result.fun < -best_score:
            candidate_params = de_result.x
            candidate_points = _params_to_points(candidate_params)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead polish on perturbation parameters
    if time.time() < deadline:
        try:
            # Start from best params found
            # Reconstruct params from best_points approximately
            # Actually, let's just use the DE result or zero
            base_angles = 2 * np.pi * np.arange(N) / N
            # Extract current params from best_points
            dx = best_points[:, 0]
            dy = best_points[:, 1]
            cur_angles = np.arctan2(dy, dx)
            # Unwrap to be near base_angles
            cur_angles = base_angles + (cur_angles - base_angles + np.pi) % (2*np.pi) - np.pi
            cur_radii = np.sqrt(dx**2 + dy**2)
            cur_params = np.zeros(2*N)
            cur_params[0::2] = cur_angles - base_angles
            cur_params[1::2] = cur_radii - 1.0
            
            nm_result = minimize(
                _negative_score_from_params,
                cur_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if nm_result.fun < -best_score:
                candidate_points = _params_to_points(nm_result.x)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Phase 3: A few random restarts with Nelder-Mead if time allows
    if time.time() < deadline - 5:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            np.random.seed(100 + restart)
            random_params = np.random.uniform(-angle_bound, angle_bound, N)
            random_radii = np.random.uniform(-0.3, 0.3, N)
            init_params = np.zeros(2*N)
            init_params[0::2] = random_params
            init_params[1::2] = random_radii
            
            try:
                nm_result = minimize(
                    _negative_score_from_params,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                if nm_result.fun < -best_score:
                    candidate_points = _params_to_points(nm_result.x)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_points = candidate_points
                        best_score = candidate_score
            except Exception:
                pass
    
    return best_points

# EVOLVE-BLOCK-END
