"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute all triangle areas vectorized. Returns array of shape (C(N,3),)."""
    n = len(pts)
    # Compute cross products for all triples
    # For points i, j, k: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

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

def _neg_soft_min_ratio(pts, temperature):
    """Negative of soft min ratio for minimization."""
    return -_soft_min_ratio(pts, temperature)

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
    # Try different ring configurations
    configs = [
        (1, 12),  # 1 center + 12 outer
        (3, 10),  # 3 inner + 10 outer
        (5, 8),   # 5 inner + 8 outer
    ]
    results = []
    for inner, outer in configs:
        pts = []
        # Inner ring
        if inner > 0:
            if inner == 1:
                pts.append([0.0, 0.0])
            else:
                inner_angles = np.linspace(0, 2 * np.pi, inner, endpoint=False)
                inner_r = 0.3
                for a in inner_angles:
                    pts.append([inner_r * np.cos(a), inner_r * np.sin(a)])
        # Outer ring
        outer_angles = np.linspace(0, 2 * np.pi, outer, endpoint=False)
        for a in outer_angles:
            pts.append([np.cos(a), np.sin(a)])
        pts = np.array(pts[:n])
        if len(pts) == n:
            results.append(pts)
    return results

def _structured_grid(n):
    """Generate structured grid-like points."""
    # Try different grid arrangements
    results = []
    # 3x5 grid (with one removed)
    for cols in range(3, 6):
        rows = n // cols
        if rows * cols < n:
            continue
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                x = (c - (cols - 1) / 2) * 0.6
                y = (r - (rows - 1) / 2) * 0.6
                pts.append([x, y])
        if len(pts) == n:
            pts = np.array(pts)
            results.append(pts)
    return results

def _generate_starts(rng):
    """Generate diverse starting configurations."""
    starts = []
    
    # Regular polygon
    starts.append(_regular_polygon(N))
    
    # Halton sequence
    starts.append(_generate_halton_points(N))
    
    # Concentric rings
    for ring_pts in _concentric_rings(N):
        starts.append(ring_pts)
    
    # Structured grids
    for grid_pts in _structured_grid(N):
        starts.append(grid_pts)
    
    # Random starts
    for _ in range(8):
        pts = rng.random((N, 2)) * 2 - 1
        pts -= np.mean(pts, axis=0)
        starts.append(pts)
    
    # Perturbed regular polygon
    base = _regular_polygon(N)
    for _ in range(5):
        noise = rng.normal(0, 0.2, size=N * 2)
        perturbed = base.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        starts.append(perturbed)
    
    return starts

def _optimize_start(start_pts, start_temp, end_temp, maxiter, time_budget):
    """Optimize from a single starting point with adaptive temperature."""
    start_time = time.time()
    best_result = start_pts.flatten().copy()
    best_score = _score(start_pts)
    
    n_steps = min(maxiter, 200)
    temps = np.linspace(start_temp, end_temp, n_steps)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        temp = temps[step]
        
        def objective(params):
            pts = params.reshape(N, 2)
            return _neg_soft_min_ratio(pts, temp)
        
        try:
            res = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': max(10, maxiter // n_steps), 'maxfun': 500, 'disp': False}
            )
            if res.success or res.fun < objective(best_result):
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result.reshape(N, 2), best_score

def _local_refinement(pts, rng, n_trials, time_budget):
    """Local refinement with random perturbations and optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for trial in range(n_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturbation
        noise_scale = 0.05 * (1.0 - trial / n_trials)
        noise = rng.normal(0, noise_scale, size=N * 2)
        perturbed = best_pts.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        
        # Optimize
        result, score = _optimize_start(perturbed, 0.01, 0.001, 500, time_budget - (time.time() - start_time))
        
        if score > best_score:
            best_score = score
            best_pts = result
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s kill limit
    
    rng = np.random.default_rng(42)
    
    best_score = -1e9
    best_pts = None
    
    # Generate diverse starts
    starts = _generate_starts(rng)
    
    # Phase 1: Coarse optimization from multiple starts
    for i, start in enumerate(starts):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10.0:
            break
        
        # Use higher temperature for early starts, lower for later
        start_temp = 0.1 if i < 5 else 0.05
        end_temp = 0.01
        maxiter = 100
        
        result, score = _optimize_start(start, start_temp, end_temp, maxiter, remaining - 20.0)
        
        if score > best_score:
            best_score = score
            best_pts = result
    
    # Phase 2: Fine optimization
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 15.0:
            result, score = _optimize_start(best_pts, 0.01, 0.001, 200, remaining - 10.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    # Phase 3: Local refinement
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10.0:
            best_pts, best_score = _local_refinement(
                best_pts, rng, n_trials=10, time_budget=remaining - 5.0
            )
    
    # Fallback
    if best_pts is None:
        return _regular_polygon(N)
    
    return best_pts

# EVOLVE-BLOCK-END
