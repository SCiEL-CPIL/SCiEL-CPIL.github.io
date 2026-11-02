import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 centrally symmetric points.
    params[0:7]: angles in (0, pi) for the 7 half-points
    params[7:14]: radii for the 7 half-points
    
    The 14 points are: p_i = r_i * (cos(a_i), sin(a_i)) and -p_i for i=0..6
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    
    # Ensure angles are in valid range
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    """Negative score for minimization."""
    points = _params_to_points(params)
    score = _score(points)
    return -score

def _objective_soft(params, beta=50.0):
    """Soft-min objective for smoother gradients."""
    points = _params_to_points(params)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 1e10
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _all_areas(points)
    if np.min(areas) < 1e-12:
        return 1e10
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    
    if soft_min < 1e-12:
        return 1e10
    
    return -(soft_min / hull_area)

def _bounds_sym():
    """
    Bounds for centrally symmetric parameters.
    """
    bounds = []
    # Angles: (0.02, pi-0.02)
    for _ in range(N_HALF):
        bounds.append((0.02, np.pi - 0.02))
    # Radii: (0.05, 3.0)
    for _ in range(N_HALF):
        bounds.append((0.05, 3.0))
    return bounds

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    bounds = _bounds_sym()
    
    # Generate initial parameter sets
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular symmetric configuration
    base_angles = np.linspace(0.1, np.pi - 0.1, N_HALF)
    base_radii = np.ones(N_HALF)
    inits.append(np.concatenate([base_angles, base_radii]))
    
    # 2. Varying radii
    for r_scale in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
        radii = np.full(N_HALF, r_scale)
        inits.append(np.concatenate([base_angles, radii]))
    
    # 3. Random perturbations
    for _ in range(30):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.2, 1.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))
    
    # 4. Larger random perturbations
    for _ in range(20):
        angles = np.sort(rng.uniform(0.05, np.pi - 0.05, N_HALF))
        radii = rng.uniform(0.1, 2.0, N_HALF)
        inits.append(np.concatenate([angles, radii]))
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=150,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.3, 1.0),
                recombination=0.7
            )
            if np.isfinite(de_result.fun):
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_params = de_result.x.copy()
                    best_points = _params_to_points(de_result.x)
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starts
    for init_params in inits:
        if time.time() > deadline - 30:
            break
        
        # Nelder-Mead
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if np.isfinite(res.fun):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = _params_to_points(res.x)
                    
        except Exception:
            continue
        
        # Powell refinement
        if best_score > 0 and time.time() < deadline - 15:
            try:
                res2 = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 1000,
                        'xtol': 1e-12,
                        'ftol': 1e-14
                    }
                )
                score2 = -res2.fun
                if np.isfinite(score2) and score2 > best_score:
                    best_score = score2
                    best_params = res2.x.copy()
                    best_points = _params_to_points(res2.x)
            except Exception:
                pass
        
        # L-BFGS-B with soft objective for smoother landscape
        if best_score > 0 and time.time() < deadline - 10:
            try:
                res3 = minimize(
                    _objective_soft,
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                if np.isfinite(res3.fun):
                    # Evaluate with hard objective
                    score3 = -_objective(res3.x)
                    if score3 > best_score:
                        best_score = score3
                        best_params = res3.x.copy()
                        best_points = _params_to_points(res3.x)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 2000,
                    'xtol': 1e-14,
                    'ftol': 1e-16
                }
            )
            if np.isfinite(res_final.fun):
                score_final = -res_final.fun
                if score_final > best_score:
                    best_score = score_final
                    best_params = res_final.x.copy()
                    best_points = _params_to_points(res_final.x)
        except Exception:
            pass
    
    # Final fallback if nothing found
    if best_points is None:
        # Return regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize to unit area
    hull_area = _compute_hull_area_fast(best_points)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale
    else:
        best_points = best_points * 1.0
        
    return best_points

def _compute_hull_area_fast(points):
    """
    Compute convex hull area.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0
