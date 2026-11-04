"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute all triangle areas vectorized. Returns array of shape (C(N,3),)."""
    n = len(pts)
    # Precompute all pairs
    areas = np.empty((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return None

def _score(pts):
    """Compute min triangle area / hull area."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_value(areas, temperature):
    """Compute soft minimum using log-sum-exp trick."""
    if len(areas) == 0:
        return 0.0
    neg_areas = -areas
    max_val = np.max(neg_areas)
    if max_val > 500:
        max_val = 500
    shifted = neg_areas - max_val
    log_sum = max_val + np.log(np.sum(np.exp(shifted)))
    return -temperature * log_sum

def _soft_min_ratio(pts, temperature):
    """Compute soft min triangle area ratio."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_value(areas, temperature)
    return soft_min / hull_area

def _neg_soft_min_ratio(params, temperature):
    """Negative of soft min ratio for minimization. params is flat array."""
    pts = params.reshape(N, 2)
    return -_soft_min_ratio(pts, temperature)

def _neg_score(params):
    """Negative of exact score for minimization."""
    pts = params.reshape(N, 2)
    return -_score(pts)

def _halton_sequence(index, base):
    """Generate Halton sequence value."""
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def _generate_halton_points(n):
    """Generate n Halton points in [0,1]^2, centered."""
    pts = []
    for i in range(1, n + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        pts.append([x, y])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    return pts

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _concentric_rings(n):
    """Generate points in concentric ring patterns."""
    configs = [
        (1, 12),
        (3, 10),
        (5, 8),
        (2, 11),
    ]
    results = []
    for inner, outer in configs:
        pts = []
        if inner > 0:
            if inner == 1:
                pts.append([0.0, 0.0])
            else:
                inner_angles = np.linspace(0, 2 * np.pi, inner, endpoint=False)
                inner_r = 0.35
                for a in inner_angles:
                    pts.append([inner_r * np.cos(a), inner_r * np.sin(a)])
        outer_angles = np.linspace(0, 2 * np.pi, outer, endpoint=False)
        for a in outer_angles:
            pts.append([np.cos(a), np.sin(a)])
        pts = np.array(pts[:n])
        if len(pts) == n:
            results.append(pts)
    return results

def _structured_grid(n):
    """Generate structured grid-like points."""
    results = []
    for cols in range(3, 6):
        rows = n // cols
        if rows * cols < n:
            continue
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                x = (c - (cols - 1) / 2) * 0.7
                y = (r - (rows - 1) / 2) * 0.7
                pts.append([x, y])
        if len(pts) == n:
            pts = np.array(pts)
            results.append(pts)
    return results

def _generate_starts(rng):
    """Generate diverse starting configurations."""
    starts = []
    starts.append(_regular_polygon(N))
    starts.append(_generate_halton_points(N))
    for ring_pts in _concentric_rings(N):
        starts.append(ring_pts)
    for grid_pts in _structured_grid(N):
        starts.append(grid_pts)
    # Random starts
    for _ in range(10):
        pts = rng.random((N, 2)) * 2 - 1
        pts -= np.mean(pts, axis=0)
        starts.append(pts)
    # Perturbed regular polygon
    base = _regular_polygon(N)
    for _ in range(6):
        noise = rng.normal(0, 0.25, size=N * 2)
        perturbed = base.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        starts.append(perturbed)
    return starts

def _lbfgs_optimize(start_pts, start_temp, end_temp, n_steps, time_budget):
    """Optimize from a single starting point with adaptive temperature."""
    start_time = time.time()
    best_result = start_pts.flatten().copy()
    best_score = _score(start_pts)
    
    temps = np.linspace(start_temp, end_temp, n_steps)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        temp = temps[step]
        
        def objective(params):
            return _neg_soft_min_ratio(params, temp)
        
        try:
            res = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 800, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result.reshape(N, 2), best_score

def _nelder_mead_polish(pts, time_budget):
    """Polish with Nelder-Mead on the exact non-smooth objective."""
    start_time = time.time()
    best_pts = pts.flatten().copy()
    best_score = _score(pts)
    
    # Multiple Nelder-Mead passes with decreasing initial simplex size
    simplex_sizes = [0.15, 0.08, 0.04, 0.02]
    
    for size in simplex_sizes:
        if time.time() - start_time > time_budget:
            break
        
        def objective(params):
            return _neg_score(params)
        
        try:
            res = minimize(
                objective,
                best_pts,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9,
                         'adaptive': True, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = res.x.copy()
        except Exception:
            pass
    
    return best_pts.reshape(N, 2), best_score

def _local_refinement(pts, rng, n_trials, time_budget):
    """Local refinement with random perturbations and optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for trial in range(n_trials):
        if time.time() - start_time > time_budget:
            break
        
        noise_scale = 0.04 * (1.0 - trial / max(n_trials, 1))
        noise = rng.normal(0, noise_scale, size=N * 2)
        perturbed = best_pts.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3.0:
            break
        
        # Quick L-BFGS then Nelder-Mead polish
        result, score = _lbfgs_optimize(perturbed, 0.01, 0.002, 5, remaining * 0.5)
        if score > best_score:
            best_score = score
            best_pts = result
        
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5.0:
            result, score = _nelder_mead_polish(best_pts, remaining - 2.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    return best_pts, best_score

def _differential_evolution_search(bounds, time_budget, rng):
    """Run differential evolution on the 2D point parameters."""
    start_time = time.time()
    
    def objective(params):
        return _neg_score(params)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=rng.integers(0, 2**31),
            maxiter=80,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            workers=1,
            update_interval=1,
            callback=None
        )
        if result.success:
            pts = result.x.reshape(N, 2)
            score = _score(pts)
            return pts, score
    except Exception:
        pass
    
    return None, -1e9

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s kill limit
    
    rng = np.random.default_rng(42)
    
    best_score = -1e9
    best_pts = None
    
    # Generate diverse starts
    starts = _generate_starts(rng)
    
    # Phase 1: L-BFGS optimization from multiple starts
    for i, start in enumerate(starts):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 8.0:
            break
        
        start_temp = 0.15 if i < 5 else 0.08
        end_temp = 0.005
        n_steps = 8
        
        result, score = _lbfgs_optimize(start, start_temp, end_temp, n_steps, remaining - 15.0)
        
        if score > best_score:
            best_score = score
            best_pts = result
    
    # Phase 2: Nelder-Mead polish on the best
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 20.0:
            result, score = _nelder_mead_polish(best_pts, remaining - 10.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    # Phase 3: Differential evolution (global search)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 30.0:
        # Bounds for each coordinate
        bounds = [(-1.5, 1.5)] * (2 * N)
        de_pts, de_score = _differential_evolution_search(bounds, remaining - 20.0, rng)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
    
    # Phase 4: Fine L-BFGS + Nelder-Mead on the best
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 15.0:
            result, score = _lbfgs_optimize(best_pts, 0.005, 0.001, 5, remaining * 0.4)
            if score > best_score:
                best_score = score
                best_pts = result
            
            remaining = time_limit - (time.time() - start_time)
            if remaining > 10.0:
                result, score = _nelder_mead_polish(best_pts, remaining - 5.0)
                if score > best_score:
                    best_score = score
                    best_pts = result
    
    # Phase 5: Local refinement
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 12.0:
            best_pts, best_score = _local_refinement(
                best_pts, rng, n_trials=8, time_budget=remaining - 5.0
            )
    
    # Final Nelder-Mead polish
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 8.0:
            result, score = _nelder_mead_polish(best_pts, remaining - 3.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    # Fallback
    if best_pts is None:
        return _regular_polygon(N)
    
    return best_pts

# EVOLVE-BLOCK-END
