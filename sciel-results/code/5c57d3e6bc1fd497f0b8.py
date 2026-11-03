"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
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


def _min_triangle_area_with_indices(points):
    """Return (min_area, indices of the min triangle)."""
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            abs_cross = np.abs(cross)
            local_min_idx = np.argmin(abs_cross)
            local_min = 0.5 * abs_cross[local_min_idx]
            if local_min < min_area:
                min_area = local_min
                min_idx = (i, j, j + 1 + local_min_idx)
                if min_area < 1e-14:
                    break
        if min_area < 1e-14:
            break
    return min_area, min_idx


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=200.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _multi_beta_soft_obj(pts_flat, betas=(40.0, 100.0, 200.0, 400.0)):
    """Multi-beta soft objective: weighted combination of soft-mins."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


def _epigraph_obj(pts_flat, t, beta=500.0):
    """
    Epigraph formulation: minimize t subject to all triangle areas >= t.
    We use a smooth penalty: t + penalty * sum(soft_min(areas - t, 0))
    Actually simpler: minimize -t + lambda * sum(max(0, t - area_i)^2)
    But let's use: objective = hull_area / t + penalty for areas < t
    """
    points = pts_flat[:2*N].reshape(N, 2)
    t_val = pts_flat[2*N]
    if np.any(~np.isfinite(points)) or t_val < 0:
        return 1e10
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    # We want to maximize t (min triangle area), so minimize -t
    # But also need t <= min(areas). Penalty for violations.
    violations = np.maximum(0, t_val - areas)
    penalty = np.sum(violations ** 2)
    return -t_val + 1000.0 * penalty


def _coordinate_descent_polish(pts, max_rounds=80, tol=1e-12):
    """Polish by adjusting points in the minimum triangle using finer steps."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    for round_num in range(max_rounds):
        min_area, (i, j, k) = _min_triangle_area_with_indices(best_pts)
        if min_area < 1e-14:
            break
        improved = False
        for pt_idx in [i, j, k]:
            for step in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        trial = best_pts.copy()
                        trial[pt_idx, 0] += dx
                        trial[pt_idx, 1] += dy
                        s = _score(trial)
                        if s > best_score + tol:
                            best_score = s
                            best_pts = trial
                            improved = True
        if not improved:
            break
    return best_pts


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    pts = []
    pts.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    mod = 1.0 + 0.15 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_perturbed(n, seed=0):
    """Regular polygon with random perturbation."""
    rng = np.random.RandomState(seed)
    base = _initial_regular_polygon(n)
    noise = rng.normal(0, 0.05, size=base.shape)
    return _normalize_unit_area(base + noise)


def construct():
    start_time = time.time()
    
    # Diverse initializations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
        _initial_perturbed(N, seed=1),
        _initial_perturbed(N, seed=2),
        _initial_perturbed(N, seed=3),
    ]
    
    best_pts = None
    best_score = 0.0
    
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        x0 = init.flatten()
        
        # Stage 1: Multi-beta soft objective (smooth, good global search)
        if time.time() - start_time < DEADLINE - 20:
            try:
                res = minimize(_multi_beta_soft_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
                    x0 = cand.flatten()
            except Exception:
                pass
        
        # Stage 2: Higher-beta soft objective (sharper min approximation)
        if time.time() - start_time < DEADLINE - 15:
            try:
                res = minimize(lambda x: _soft_obj(x, beta=500.0), x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
                    x0 = cand.flatten()
            except Exception:
                pass
        
        # Stage 3: Hard objective (exact min triangle area)
        if time.time() - start_time < DEADLINE - 10:
            try:
                res = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                      'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
    
    # Final high-precision polish with hard objective
    if best_pts is not None and time.time() - start_time < DEADLINE - 12:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-13,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Coordinate descent polish on the minimum triangle
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            polished = _coordinate_descent_polish(best_pts, max_rounds=50, tol=1e-13)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_pts = polished
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
