"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def _build_points_from_params(params):
    """
    Build 14 points from 28 parameters: 14 angles + 14 radii.
    Angles are in [0, 2*pi), sorted to ensure convex position.
    Radii are positive values controlling distance from origin.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Sort angles and corresponding radii together
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]
    
    # Ensure angles are in [0, 2*pi)
    angles_sorted = angles_sorted % (2 * np.pi)
    
    # Build points
    points = np.column_stack([
        radii_sorted * np.cos(angles_sorted),
        radii_sorted * np.sin(angles_sorted)
    ])
    
    return points

def _min_triangle_area_fast(points):
    """Fast vectorized minimum triangle area computation."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            # Vectorized over k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    
    return min_area

def _hull_area_shoelace(points):
    """Compute convex hull area using shoelace formula (assumes convex order)."""
    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    return area

def _score_from_params(params):
    """Compute score from 28 parameters (14 angles + 14 radii)."""
    points = _build_points_from_params(params)
    
    # Check for degenerate cases
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 0.0
    
    # Compute hull area (points should be in convex order due to sorted angles)
    hull_area = _hull_area_shoelace(points)
    if hull_area < 1e-10:
        return 0.0
    
    # Compute min triangle area
    min_area = _min_triangle_area_fast(points)
    if min_area < 1e-12:
        return 0.0
    
    return min_area / hull_area

def _objective(params):
    """
    Objective function to minimize: 1/score = hull_area / min_triangle_area.
    Uses a soft penalty to handle invalid configurations.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Check for near-duplicate angles
    sorted_angles = np.sort(angles % (2 * np.pi))
    diffs = np.diff(sorted_angles)
    diffs = np.append(diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
    if np.min(diffs) < 1e-6:
        return 1e8 + (1e-6 - np.min(diffs)) * 1e6
    
    # Check radii are positive
    if np.any(radii < 0.01):
        return 1e8 + (0.01 - np.min(radii)) * 1e6
    
    points = _build_points_from_params(params)
    
    hull_area = _hull_area_shoelace(points)
    if hull_area < 1e-10:
        return 1e10
    
    min_area = _min_triangle_area_fast(points)
    if min_area < 1e-12:
        return 1e10
    
    return hull_area / min_area

def _normalize_unit_area(points):
    """Scale points so convex hull area = 1."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def construct():
    """
    Generate 14 points to maximize minimum triangle area / convex hull area.
    
    Strategy: Use dual parameterization (angles + radii) for convex position.
    Points are placed at (r_i * cos(theta_i), r_i * sin(theta_i)) where theta_i are
    sorted angles and r_i > 0 are radii. This allows non-circular convex shapes.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_params = None
    
    # Initial guess: equally spaced angles, unit radii
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_radii = np.ones(N)
    base_params = np.concatenate([base_angles, base_radii])
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Differential Evolution for global search ---
    # Define bounds for angles and radii
    angle_bounds = [(0, 2*np.pi)] * N
    radius_bounds = [(0.5, 2.0)] * N
    bounds = angle_bounds + radius_bounds
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mut=(0.3, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                update_interval=1
            )
            
            if de_result.fun < 1e7:
                score_val = _score_from_params(de_result.x)
                if score_val > best_score:
                    best_score = score_val
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # --- Phase 2: Multi-start Nelder-Mead refinement ---
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0 and best_params is not None:
            x0 = best_params.copy()
        elif restart == 1:
            x0 = base_params.copy()
        else:
            # Random perturbation
            if best_params is not None:
                x0 = best_params + rng.normal(0, 0.1, size=2*N)
            else:
                x0 = base_params + rng.normal(0, 0.2, size=2*N)
            x0[:N] = x0[:N] % (2*np.pi)
            x0[N:] = np.clip(x0[N:], 0.5, 2.0)
        
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res.fun < 1e7:
                score_val = _score_from_params(res.x)
                if score_val > best_score:
                    best_score = score_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # --- Phase 3: Powell polish on best params ---
    if best_params is not None and time.time() < deadline - 20:
        try:
            res_powell = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 3000,
                    'xtol': 1e-12,
                    'ftol': 1e-14
                }
            )
            if res_powell.fun < 1e7:
                score_val = _score_from_params(res_powell.x)
                if score_val > best_score:
                    best_score = score_val
                    best_params = res_powell.x.copy()
        except Exception:
            pass
    
    # --- Phase 4: Final NM polish ---
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res_final.fun < 1e7:
                score_val = _score_from_params(res_final.x)
                if score_val > best_score:
                    best_score = score_val
                    best_params = res_final.x.copy()
        except Exception:
            pass
    
    # --- Convert params to points ---
    if best_params is None:
        # Fallback: regular 14-gon
        best_params = base_params.copy()
    
    final_points = _build_points_from_params(best_params)
    
    # Normalize to unit area
    final_points = _normalize_unit_area(final_points)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            fallback_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            final_points = np.column_stack([np.cos(fallback_angles), np.sin(fallback_angles)])
            final_points = _normalize_unit_area(final_points)
    except Exception:
        fallback_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(fallback_angles), np.sin(fallback_angles)])
        final_points = _normalize_unit_area(final_points)
    
    return final_points
# EVOLVE-BLOCK-END
