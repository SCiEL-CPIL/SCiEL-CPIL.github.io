"""Concentric ring packing: discrete radial shells + smooth gradient optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return _DEADLINE - _elapsed()


def _compute_radius(points):
    """Compute the maximum valid packing radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between pairs
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Min distance to boundary
    dists = 1.0 - np.sqrt(np.sum(points ** 2, axis=1))
    min_bound = dists.min()
    return min(min_pair / 2.0, min_bound)


def _soft_min(vals, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(vals)
    return m - np.log(np.sum(np.exp(-beta * (vals - m)))) / beta


def _ring_points(radii, phases, counts):
    """Generate points from ring structure."""
    pts = []
    for r, ph, c in zip(radii, phases, counts):
        if c == 1:
            pts.append([r * np.cos(ph), r * np.sin(ph)])
        else:
            angles = ph + 2.0 * np.pi * np.arange(c) / c
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    return np.vstack(pts)


def _objective(params, counts, radii_bounds):
    """Objective: maximize soft-min of constraints (minimize negative)."""
    n_params = len(counts)
    radii = params[:n_params]
    phases = params[n_params:]
    
    # Clip radii to valid range
    radii = np.clip(radii, 0.0, 1.0)
    
    pts = _ring_points(radii, phases, counts)
    if len(pts) != N:
        return 1e6
    
    # Check all points are inside unit circle
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    if np.any(dists > 1.0):
        return 1e6
    
    # Compute pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # Soft constraints
    min_pair = _soft_min(d.ravel(), beta=30.0) / 2.0
    min_bound = _soft_min(1.0 - dists, beta=30.0)
    
    return -min(min_pair, min_bound)


def _optimize_ring(counts, radii_init, phases_init, max_iters=200):
    """Optimize a given ring configuration."""
    if _remaining() < 2.0:
        return None
    
    n_params = len(counts)
    x0 = np.concatenate([radii_init, phases_init])
    
    def obj(x):
        return _objective(x, counts, None)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iters, 'gtol': 1e-8})
        if _remaining() < 1.0:
            return None
        best_params = res.x
        r = -obj(best_params)
        if r <= 0:
            return None
        return r, best_params
    except Exception:
        return None


def _refine_continuous(points, max_iters=50):
    """Final continuous refinement of the best ring solution."""
    if _remaining() < 3.0:
        return points
    
    n = len(points)
    x0 = points.ravel()
    
    def obj(x):
        pts = x.reshape(n, 2)
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        dists = np.sqrt(np.sum(pts ** 2, axis=1))
        min_pair = _soft_min(d.ravel(), beta=40.0) / 2.0
        min_bound = _soft_min(1.0 - dists, beta=40.0)
        return -min(min_pair, min_bound)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'gtol': 1e-9})
        if _remaining() < 0.5:
            return points
        new_pts = res.x.reshape(n, 2)
        if _compute_radius(new_pts) > _compute_radius(points):
            return new_pts
    except Exception:
        pass
    return points


def _generate_ring_configs():
    """Generate diverse ring count configurations summing to N."""
    configs = []
    
    # Configuration 1: Dense center, expanding rings
    # 1 + 6 + 12 + 18 + 13 = 50
    configs.append([1, 6, 12, 18, 13])
    
    # Configuration 2: More uniform
    # 1 + 6 + 12 + 15 + 16 = 50
    configs.append([1, 6, 12, 15, 16])
    
    # Configuration 3: Fewer, larger rings
    # 1 + 6 + 12 + 18 + 13 (same as 1, try different)
    # 7 + 12 + 18 + 13 = 50
    configs.append([7, 12, 18, 13])
    
    # Configuration 4: 6 rings
    # 1 + 5 + 10 + 14 + 12 + 8 = 50
    configs.append([1, 5, 10, 14, 12, 8])
    
    # Configuration 5: 1 + 6 + 12 + 18 + 13 but shifted
    # 0 + 6 + 12 + 18 + 14 = 50 (no center point, outer ring 14)
    configs.append([6, 12, 18, 14])
    
    # Configuration 6: 1 + 6 + 12 + 18 + 13
    # Try 1 + 6 + 12 + 17 + 14 = 50
    configs.append([1, 6, 12, 17, 14])
    
    # Configuration 7: Hexagonal-inspired
    # 1 + 6 + 12 + 18 + 13
    configs.append([1, 6, 12, 18, 13])
    
    # Configuration 8: 1 + 6 + 12 + 18 + 13 with different split
    # 1 + 6 + 11 + 18 + 14 = 50
    configs.append([1, 6, 11, 18, 14])
    
    # Configuration 9: 2 + 6 + 12 + 18 + 12 = 50
    configs.append([2, 6, 12, 18, 12])
    
    # Configuration 10: 1 + 6 + 12 + 18 + 13 (duplicate check)
    # 3 + 6 + 12 + 18 + 11 = 50
    configs.append([3, 6, 12, 18, 11])
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    return unique_configs


def construct():
    global _start_time
    _start_time = time.time()
    
    best_radius = 0.0
    best_points = None
    
    configs = _generate_ring_configs()
    
    for counts in configs:
        if _remaining() < 5.0:
            break
        if sum(counts) != N:
            continue
        
        n_rings = len(counts)
        
        # Initialize radii: evenly spaced
        radii_init = np.linspace(0.15, 0.85, n_rings)
        
        # Try multiple phase initializations
        n_phase_trials = 3
        for trial in range(n_phase_trials):
            if _remaining() < 3.0:
                break
            
            if trial == 0:
                phases_init = np.zeros(n_rings)
            elif trial == 1:
                phases_init = np.pi / (2 * n_rings) * np.arange(n_rings)
            else:
                phases_init = np.pi * np.random.default_rng(trial).uniform(0, 1, n_rings)
            
            result = _optimize_ring(counts, radii_init, phases_init)
            if result is None:
                continue
            
            r, params = result
            if r > best_radius:
                best_radius = r
                # Reconstruct points
                radii = params[:n_rings]
                phases = params[n_rings:]
                best_points = _ring_points(radii, phases, counts)
    
    if best_points is None or _remaining() < 2.0:
        # Fallback: hexagonal lattice
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    # Final continuous refinement
    best_points = _refine_continuous(best_points)
    
    # Ensure validity
    final_r = _compute_radius(best_points)
    if final_r <= 0 or not np.all(np.isfinite(best_points)):
        # Emergency fallback
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points
