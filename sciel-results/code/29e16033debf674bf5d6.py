"""Program for Heilbronn convex regions n=14 using polar coordinates with sorted angles and differential evolution."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

DEADLINE = 110.0  # Safety margin before 120s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a


def _params_to_points(params):
    """
    Convert parameters to Cartesian coordinates.
    params: [r_0, ..., r_{N-1}, dtheta_1, ..., dtheta_{N-1}]
    theta_0 is fixed at 0.
    theta_i = sum(dtheta_1..dtheta_i)
    """
    n = N
    radii = params[:n]
    dthetas = params[n:]  # N-1 values
    
    # Build angles: theta_0 = 0, theta_i = cumsum(dthetas[:i])
    angles = np.zeros(n)
    if n > 1:
        angles[1:] = np.cumsum(dthetas)
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for minimization: negative of score."""
    points = _params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_a = _hull_area(points)
    if hull_a <= 0:
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-14:
        return 1e10
    return -min_area / hull_a


def _initial_params_regular():
    """Generate initial polar parameters for a regular polygon."""
    R = 1.0
    radii = np.ones(N) * R
    dthetas = np.full(N - 1, 2.0 * np.pi / N)
    return np.concatenate([radii, dthetas])


def _initial_params_random(seed):
    """Generate random initial polar parameters."""
    rng = np.random.RandomState(seed)
    radii = rng.uniform(0.5, 1.5, N)
    base_dtheta = 2.0 * np.pi / N
    dthetas = base_dtheta * (1.0 + 0.3 * rng.randn(N - 1))
    dthetas = np.clip(dthetas, 0.05, base_dtheta * 1.5)
    return np.concatenate([radii, dthetas])


def _local_refine(params, max_iter=200):
    """Local refinement using L-BFGS-B, with Nelder-Mead fallback."""
    n = N
    r_bounds = [(0.1, 2.0)] * n
    max_dtheta = (2.0 * np.pi / N) * 1.3
    min_dtheta = 0.02
    dtheta_bounds = [(min_dtheta, max_dtheta)] * (n - 1)
    bounds = r_bounds + dtheta_bounds
    
    best_params = params
    best_obj = _objective(params)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Try Nelder-Mead as fallback if L-BFGS-B didn't improve much
    if time.time() - _start_time < DEADLINE - 10:
        try:
            nm_params = best_params.copy()
            result_nm = minimize(
                _objective,
                nm_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result_nm.fun < best_obj:
                best_obj = result_nm.fun
                best_params = result_nm.x
        except Exception:
            pass
    
    return best_params


def _polish(params, time_budget=10.0):
    """Final polish: try small perturbations around the best solution."""
    start_t = time.time()
    best_params = params.copy()
    best_score = _score(_params_to_points(params))
    
    rng = np.random.RandomState(42)
    for _ in range(200):
        if time.time() - start_t > time_budget:
            break
        # Random small perturbation
        scale = 0.05
        perturbation = rng.randn(len(params)) * scale
        cand = best_params + perturbation
        # Keep within reasonable bounds
        cand[:N] = np.clip(cand[:N], 0.1, 2.0)
        cand[N:] = np.clip(cand[N:], 0.02, (2.0 * np.pi / N) * 1.3)
        
        s = _score(_params_to_points(cand))
        if s > best_score:
            best_score = s
            best_params = cand
    
    return best_params


_start_time = None


def construct():
    global _start_time
    _start_time = time.time()
    
    min_dtheta = 0.03
    max_dtheta = 2.0 * np.pi / N * 1.4
    
    r_bounds = [(0.1, 2.0)] * N
    dtheta_bounds = [(min_dtheta, max_dtheta)] * (N - 1)
    bounds = r_bounds + dtheta_bounds
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Differential evolution with multiple diverse starts
    seeds = [42, 123, 456, 789]
    for seed in seeds:
        if time.time() - _start_time > DEADLINE - 30:
            break
        try:
            if seed == 42:
                x0 = _initial_params_regular()
            else:
                x0 = _initial_params_random(seed)
            
            result = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0,
                seed=seed,
                maxiter=100,
                tol=1e-8,
                popsize=8,
                polish=False,
                init='sobol'
            )
            if result.x is not None:
                cand_pts = _params_to_points(result.x)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = result.x
        except Exception:
            pass
    
    # Phase 2: Local refinement of best candidate
    if best_params is not None and time.time() - _start_time < DEADLINE - 15:
        try:
            refined = _local_refine(best_params, max_iter=300)
            cand_pts = _params_to_points(refined)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = refined
        except Exception:
            pass
    
    # Phase 3: Second round of DE from refined point if time allows
    if time.time() - _start_time < DEADLINE - 20:
        try:
            x0_2 = best_params if best_params is not None else _initial_params_regular()
            result2 = differential_evolution(
                _objective,
                bounds=bounds,
                x0=x0_2,
                seed=999,
                maxiter=80,
                tol=1e-9,
                popsize=6,
                polish=False,
                init='sobol'
            )
            if result2.x is not None:
                refined2 = _local_refine(result2.x, max_iter=200)
                cand_pts = _params_to_points(refined2)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_params = refined2
        except Exception:
            pass
    
    # Phase 4: Final polish
    if best_params is not None and time.time() - _start_time < DEADLINE - 5:
        try:
            polished = _polish(best_params, time_budget=min(8.0, DEADLINE - (time.time() - _start_time) - 2))
            cand_pts = _params_to_points(polished)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                best_params = polished
        except Exception:
            pass
    
    if best_params is None:
        best_params = _initial_params_regular()
        
    return _params_to_points(best_params)
# EVOLVE-BLOCK-END
