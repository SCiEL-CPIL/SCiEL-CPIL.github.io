import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 585.0  # Safety margin before 600s limit

# Precompute all C(13,3) = 286 combinations
_COMBOS = np.array(list(itertools.combinations(range(13), 3)))
_N_COMBOS = len(_COMBOS)

def _triangle_areas(points):
    """Compute areas of all 286 triangles. Returns array of shape (286,)."""
    a = points[_COMBOS[:, 0]]
    b = points[_COMBOS[:, 1]]
    c = points[_COMBOS[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return areas

def _min_triangle_area(points):
    return float(np.min(_triangle_areas(points)))

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0], atol=1e-10) or np.allclose(points[:, 1], points[0, 1], atol=1e-10):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _ratio(points):
    """Compute min_triangle_area / hull_area. Returns 0 if invalid."""
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma < 1e-12:
        return 0.0
    return ma / ha

def _soft_min_ratio(params, temperature):
    """Smooth surrogate: soft-min of triangle areas / hull area.
    Returns negative value to minimize."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    if not np.all(np.isfinite(points)):
        return 1e10
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10
    
    areas = _triangle_areas(points)
    
    # Soft min via log-sum-exp trick
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_a - temperature * log_sum
    
    if soft_min < 1e-12:
        return 1e10
    
    return -(soft_min / ha)

def _exact_neg_ratio(params):
    """Exact objective for final refinement."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    r = _ratio(points)
    if r < 1e-12:
        return 1e10
    return -r

def _repulsive_energy(params, k=100.0, r_min=0.05):
    """Thomson-like repulsive energy: sum of 1/d^k over all pairs, with floor on distance.
    This encourages spreading points apart, which correlates with larger min triangle area."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # avoid self-interaction
    dists = np.maximum(dists, r_min)  # floor to avoid singularity
    
    # Repulsive energy
    energy = np.sum(1.0 / (dists ** k))
    
    # Also penalize if hull area is too small (degenerate)
    ha = _hull_area(points)
    if ha < 1e-6:
        return 1e10
    
    return float(energy)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_rings(n_inner, n_outer, r_inner=0.3, r_outer=1.0, angle_offset=0.0):
    """Points on two concentric circles with optional angle offset for inner ring."""
    inner_angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _generate_structured_configs(rng):
    """Generate diverse structured initial configurations."""
    configs = []
    
    # Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))
    
    # Concentric rings with various ratios and offsets
    for n_in, n_out, r_in, offset in [
        (6, 7, 0.35, 0.0),
        (6, 7, 0.40, np.pi/7),
        (6, 7, 0.45, np.pi/14),
        (5, 8, 0.35, 0.0),
        (5, 8, 0.40, np.pi/8),
        (4, 9, 0.30, 0.0),
        (4, 9, 0.35, np.pi/9),
        (7, 6, 0.30, 0.0),
        (7, 6, 0.35, np.pi/6),
        (3, 10, 0.25, 0.0),
        (3, 10, 0.30, np.pi/10),
        (2, 11, 0.20, 0.0),
        (8, 5, 0.50, 0.0),
    ]:
        configs.append(_concentric_rings(n_in, n_out, r_in, 1.0, offset))
    
    # Perturbed versions of best topologies
    for _ in range(6):
        base = _concentric_rings(6, 7, 0.38, 1.0, np.pi/7)
        perturbed = base + rng.normal(0, 0.04, (13, 2))
        configs.append(perturbed)
    
    return configs

def _refine_with_soft_min(params, temps=[0.5, 0.1, 0.02], max_iters_coarse=150, max_iters_fine=80):
    """Coarse-to-fine optimization: soft-min surrogate then exact."""
    if time.time() > _DEADLINE - 30:
        return params
    
    # Coarse phases with decreasing temperature
    for temp in temps:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(lambda p: _soft_min_ratio(p, temp), params,
                          method='Powell',
                          options={'maxiter': max_iters_coarse, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.x is not None and np.all(np.isfinite(res.x)):
                params = res.x
        except Exception:
            continue
    
    # Fine phase: exact objective
    if time.time() > _DEADLINE - 15:
        return params
    try:
        res = minimize(_exact_neg_ratio, params,
                      method='Powell',
                      options={'maxiter': max_iters_fine, 'xtol': 1e-10, 'ftol': 1e-10})
        if res.x is not None and np.all(np.isfinite(res.x)):
            params = res.x
    except Exception:
        pass
    
    return params

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Generate structured configs and evaluate
    configs = _generate_structured_configs(rng)
    
    for config in configs:
        if time.time() > _DEADLINE - 120:
            break
        
        init_score = _ratio(config)
        if init_score > best_score:
            best_score = init_score
            best_points = config.copy()
    
    # Phase 2: Coarse-to-fine optimization on the best structured config
    if best_points is not None:
        params = best_points.flatten()
        params = _refine_with_soft_min(params, temps=[0.5, 0.1, 0.02], max_iters_coarse=150, max_iters_fine=80)
        opt_points = params.reshape((n, 2))
        opt_score = _ratio(opt_points)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Phase 3: Multi-start refinement from other structured configs
    for config in configs[:10]:
        if time.time() > _DEADLINE - 30:
            break
        
        params = config.flatten()
        
        # Quick soft-min optimization
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(lambda p: _soft_min_ratio(p, 0.1), params,
                          method='Powell',
                          options={'maxiter': 80, 'xtol': 1e-8, 'ftol': 1e-8})
            params = res.x
            
            # Exact refinement
            res = minimize(_exact_neg_ratio, params,
                          method='Powell',
                          options={'maxiter': 40, 'xtol': 1e-10, 'ftol': 1e-10})
            opt_points = res.x.reshape((n, 2))
            opt_score = _ratio(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            continue
    
    # Phase 4: Thomson-problem-inspired search
    # Use repulsive energy to find well-spread configurations, then refine
    if time.time() < _DEADLINE - 60:
        for trial in range(5):
            if time.time() > _DEADLINE - 30:
                break
            
            # Start from a perturbed regular polygon or concentric ring
            if trial % 2 == 0:
                start = _regular_polygon(13, radius=1.0) + rng.normal(0, 0.1, (n, 2))
            else:
                start = _concentric_rings(6, 7, 0.4, 1.0, np.pi/7) + rng.normal(0, 0.1, (n, 2))
            
            params = start.flatten()
            
            # Optimize repulsive energy (spread points apart)
            try:
                res = minimize(_repulsive_energy, params,
                              method='Powell',
                              options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6})
                params = res.x
            except Exception:
                continue
            
            # Then refine with soft-min + exact
            params = _refine_with_soft_min(params, temps=[0.1, 0.02], max_iters_coarse=60, max_iters_fine=40)
            opt_points = params.reshape((n, 2))
            opt_score = _ratio(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
    
    # Phase 5: Final tight refinement of the best solution
    if best_points is not None and best_score > 0.01 and time.time() < _DEADLINE - 20:
        params = best_points.flatten()
        try:
            res = minimize(_exact_neg_ratio, params,
                          method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12})
            opt_points = res.x.reshape((n, 2))
            opt_score = _ratio(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Fallback to regular polygon
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, radius=1.0)
        best_score = _ratio(best_points)
    
    # Final check: Ensure points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points
