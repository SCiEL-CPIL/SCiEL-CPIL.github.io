"""Heilbronn triangle problem: 11 points in unit-area equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

def _min_area_bary(points_bary):
    """Compute minimum triangle area for points given in barycentric coords (u, v) where w=1-u-v.
    points_bary: (N, 2) array.
    Returns the minimum area (in units of the triangle area).
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k]) 
                     - v[i] * (u[j] * w[k] - w[j] * u[k]) 
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                a = abs(det)
                if a < min_area:
                    min_area = a
    return min_area

def _soft_min_obj(params, temperature=0.01):
    """Soft-min objective: smooth approximation of negative min area."""
    points_bary = params.reshape(N, 2)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(w < -1e-8):
        return 1e10
    
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    w = np.maximum(w, 0)
    
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k]) 
                     - v[i] * (u[j] * w[k] - w[j] * u[k]) 
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                areas.append(abs(det))
    
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-areas/T) / T approximates min(areas)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp * temperature
    return -soft_min_val

def _hard_min_obj(params):
    """Hard min objective: negative of min area."""
    points_bary = params.reshape(N, 2)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(w < -1e-8):
        return 1e10
    
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    w = np.maximum(w, 0)
    
    min_a = _min_area_bary(np.column_stack([u, v]))
    return -min_a

def _bary_to_cart(points_bary):
    """Convert barycentric coords (u, v, w) to Cartesian in the unit equilateral triangle."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    h = np.sqrt(3) / 2
    x = v * 1.0 + w * 0.5
    y = w * h
    return np.column_stack([x, y])

def _generate_hex_init():
    """Generate a hexagonal-like initial configuration in barycentric coords."""
    points = []
    # Place points in a roughly hexagonal pattern
    # Center
    points.append([1/3, 1/3])
    # Ring 1: 6 points
    for angle in range(0, 360, 60):
        r = 0.15
        rad = np.radians(angle)
        # In barycentric, we need to map polar to triangle
        # Use a simple mapping: project onto triangle
        u = 1/3 + r * np.cos(rad) * 0.5
        v = 1/3 + r * np.sin(rad) * 0.5
        w = 1 - u - v
        if w >= 0:
            points.append([u, v])
        else:
            # Adjust to be inside
            u = 1/3 + r * 0.3 * np.cos(rad)
            v = 1/3 + r * 0.3 * np.sin(rad)
            w = 1 - u - v
            points.append([u, v])
    # Need 11 total, have 7, add 4 more
    for angle in range(0, 360, 90):
        r = 0.3
        rad = np.radians(angle)
        u = 1/3 + r * np.cos(rad) * 0.4
        v = 1/3 + r * np.sin(rad) * 0.4
        w = 1 - u - v
        if w >= 0:
            points.append([u, v])
        else:
            u = 1/3 + r * 0.2 * np.cos(rad)
            v = 1/3 + r * 0.2 * np.sin(rad)
            w = 1 - u - v
            points.append([u, v])
    
    points = np.array(points[:N])
    return points

def _generate_random_init(rng):
    """Generate random initial points in barycentric coords."""
    bary = rng.dirichlet([1, 1, 1], size=N)
    return np.column_stack([bary[:, 0], bary[:, 1]])

def _generate_perturbed_init(base_points, rng, sigma=0.05):
    """Generate perturbed version of base points."""
    points = base_points + rng.normal(0, sigma, size=base_points.shape)
    u = np.maximum(points[:, 0], 0)
    v = np.maximum(points[:, 1], 0)
    w = 1 - u - v
    # Clamp to valid region
    for i in range(N):
        if w[i] < 0:
            # Push back inside
            scale = (1 - 1e-6) / (u[i] + v[i])
            u[i] *= scale
            v[i] *= scale
    return np.column_stack([u, v])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    h = np.sqrt(3) / 2
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_points_bary = None
    best_score = -float('inf')
    
    def _evaluate_and_update(points_bary):
        nonlocal best_points_bary, best_score
        u = np.maximum(points_bary[:, 0], 0)
        v = np.maximum(points_bary[:, 1], 0)
        w = 1.0 - u - v
        if np.any(w < -1e-8):
            return
        u = np.maximum(u, 0)
        v = np.maximum(v, 0)
        w = np.maximum(w, 0)
        s = u + v + w
        if np.any(s < 1e-10):
            return
        u /= s
        v /= s
        points_bary = np.column_stack([u, v])
        score = _min_area_bary(points_bary)
        if score > best_score:
            best_score = score
            best_points_bary = points_bary.copy()
    
    # Stage 1: Soft-min optimization from multiple starts
    rng = np.random.RandomState(42)
    
    # Initial configurations
    inits = [_generate_hex_init()]
    for _ in range(8):
        inits.append(_generate_random_init(rng))
    
    for init_idx, init_bary in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        x0 = init_bary.flatten()
        
        # Stage 1a: Soft-min with L-BFGS-B
        try:
            result_soft = minimize(
                _soft_min_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            _evaluate_and_update(result_soft.x.reshape(N, 2))
        except Exception:
            pass
        
        # Stage 1b: Hard min with L-BFGS-B from soft-min result
        if time.time() > deadline - 15:
            break
        
        try:
            result_hard = minimize(
                _hard_min_obj,
                result_soft.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            _evaluate_and_update(result_hard.x.reshape(N, 2))
        except Exception:
            pass
    
    # Stage 2: Local refinement with Nelder-Mead from best
    if best_points_bary is not None and time.time() < deadline - 10:
        for nm_restart in range(5):
            if time.time() > deadline - 5:
                break
            
            x0 = best_points_bary.flatten()
            if nm_restart > 0:
                x0 = x0 + rng.normal(0, 0.01, size=x0.shape)
            
            try:
                nm_result = minimize(
                    _hard_min_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
                )
                _evaluate_and_update(nm_result.x.reshape(N, 2))
            except Exception:
                pass
    
    # Fallback
    if best_points_bary is None:
        grid_points = []
        for i in range(4):
            for j in range(4 - i):
                u = j / 3.0
                v = i / 3.0
                w = 1 - u - v
                if w >= -1e-10:
                    grid_points.append([u, v])
        grid_points.append([1/3, 1/3])
        best_points_bary = np.array(grid_points[:N])
    
    # Convert to Cartesian
    points_cart = _bary_to_cart(best_points_bary)
    
    # Final validation: ensure points are inside the triangle
    for i in range(N):
        x, y = points_cart[i]
        eps = 1e-9
        y = max(eps, y)
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x - eps
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x) - eps
        x = max(eps, min(1 - eps, x))
        points_cart[i] = [x, y]
    
    return points_cart
