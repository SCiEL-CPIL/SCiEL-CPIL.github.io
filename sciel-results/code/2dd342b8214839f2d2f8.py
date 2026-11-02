"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _soft_penalty(val, beta=100.0):
    """Smooth soft penalty: approximates max(0, val)^2 with smooth function."""
    # Use log-sum-exp style: for val > 0, ~val^2; for val < 0, ~0
    # Actually simpler: use (log(1 + exp(beta*val)) - log(2)) * 2 / beta for softplus, then square
    # But simpler: just use val^2 when val > 0, with smooth transition
    # Let's use: 0.5 * (sqrt(val^2 + eps) - eps) * 2 for soft abs, then square
    # Simplest effective approach: clip and square, but add tiny epsilon for smoothness
    if val > 0:
        return val * val
    else:
        # Smooth transition: exp(-beta * |val|) * val^2 / (1 + exp(-beta * |val|))
        # For simplicity, use: 0.5 * (1 - np.exp(-beta * abs(val))) * val * val
        # But this is still not smooth at 0. Let's just use val^2 always but scale down
        return 0.0

def _objective_delaunay(params, edges_cache, penalty_weight=50.0):
    """
    Objective using Delaunay-based constraints with smooth penalty.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties - use smooth version
    for i in range(N):
        v = x[i] - r[i]
        if v < 0:
            penalty += v * v
        v = x[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        v = y[i] - r[i]
        if v < 0:
            penalty += v * v
        v = y[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    edges_set = set(edges)
    
    # Overlap penalties only for Delaunay neighbors
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap * overlap
    
    # Also check nearby non-Delaunay pairs
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edges_set:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            max_r = max(r[i], r[j])
            if dist_sq < (4 * max_r * max_r):
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap * overlap
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight=100.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        v = x[i] - r[i]
        if v < 0:
            penalty += v * v
        v = x[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        v = y[i] - r[i]
        if v < 0:
            penalty += v * v
        v = y[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap * overlap
    
    return -np.sum(r) + penalty_weight * penalty

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_dense_center():
    """Dense center with smaller outer circles."""
    positions = []
    # Central large circle
    positions.append([0.5, 0.5, 0.12])
    # Ring of 6 around center
    for k in range(6):
        angle = 2 * np.pi * k / 6
        x = 0.5 + 0.15 * np.cos(angle)
        y = 0.5 + 0.15 * np.sin(angle)
        positions.append([x, y, 0.09])
    # Ring of 12 further out
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        positions.append([x, y, 0.07])
    # Ring of 7 at corners/edges
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.38 * np.cos(angle)
        y = 0.5 + 0.38 * np.sin(angle)
        positions.append([x, y, 0.055])
    return np.array(positions[:N])

def _initial_config_mixed():
    """Mixed sizes: a few large, many small."""
    positions = []
    # 3 large circles in triangle
    positions.append([0.3, 0.3, 0.11])
    positions.append([0.7, 0.3, 0.11])
    positions.append([0.5, 0.7, 0.11])
    # 8 medium circles in ring
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.28 * np.cos(angle)
        y = 0.5 + 0.28 * np.sin(angle)
        positions.append([x, y, 0.065])
    # 15 small circles scattered
    np.random.seed(123)
    for k in range(15):
        x = np.random.uniform(0.08, 0.92)
        y = np.random.uniform(0.08, 0.92)
        r = np.random.uniform(0.03, 0.05)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _optimize_delaunay(xyz, time_budget=30.0):
    """
    Optimize using multi-phase approach:
    1. Low penalty weight, find good geometry
    2. Medium penalty, tighten constraints
    3. High penalty + full pairwise, final polish
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    edges_cache = {'edges': None}
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty weight, focus on maximizing radii
    try:
        result = minimize(lambda p: _objective_delaunay(p, edges_cache, 10.0),
                          best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < _objective_delaunay(best_params, edges_cache, 10.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.4:
        return best_params.reshape(N, 3)
    
    # Phase 2: Medium penalty weight
    try:
        result = minimize(lambda p: _objective_delaunay(p, edges_cache, 50.0),
                          best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_delaunay(best_params, edges_cache, 50.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.7:
        return best_params.reshape(N, 3)
    
    # Phase 3: High penalty, full pairwise for final polish
    try:
        result = minimize(lambda p: _objective_full(p, 200.0),
                          best_params, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < _objective_full(best_params, 200.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Dense center pattern
    initial_configs.append(_initial_config_dense_center())
    
    # 4. Mixed sizes
    initial_configs.append(_initial_config_mixed())
    
    # 5. Random configurations with different seeds
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 6. Corner-biased random
    np.random.seed(789)
    x = np.concatenate([
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.7, 0.95, 7),
        np.random.uniform(0.05, 0.3, 6),
        np.random.uniform(0.7, 0.95, 6)
    ])
    y = np.concatenate([
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.7, 0.95, 6),
        np.random.uniform(0.7, 0.95, 6)
    ])
    r = np.random.uniform(0.06, 0.12, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    n_configs = len(initial_configs)
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 8.0:
            break
        
        # Allocate more time to later configs (they may be better)
        time_budget = min(20.0, remaining - 3.0)
        if time_budget < 3.0:
            break
        
        xyz_opt = _optimize_delaunay(xyz0, time_budget=time_budget)
        
        # Fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_config(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation and fix
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
