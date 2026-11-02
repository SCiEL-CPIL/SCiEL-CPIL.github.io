"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _get_nearby_pairs(x, y, r, threshold=0.3):
    """Get pairs of circles that are close enough to potentially overlap."""
    n = len(x)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            max_r = max(r[i], r[j])
            if dist_sq < (2.0 * max_r + threshold) ** 2:
                pairs.append((i, j))
    return pairs

def _objective_nearby(params, penalty_weight=50.0, threshold=0.3):
    """
    Objective using nearby-pairs constraints (avoids expensive Delaunay).
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties (vectorized)
    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Overlap penalties for nearby pairs
    pairs = _get_nearby_pairs(x, y, r, threshold)
    for i, j in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight=100.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties (vectorized)
    pen_x_lo = np.maximum(0, -(x - r))
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, -(y - r))
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Full pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + penalty_weight * penalty

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

def _initial_config_force_directed():
    """
    Force-directed initial configuration:
    Start with random positions, apply repulsion forces to spread out,
    then set radii based on local density.
    """
    np.random.seed(123)
    x = np.random.uniform(0.15, 0.85, N)
    y = np.random.uniform(0.15, 0.85, N)
    
    # Apply repulsion to spread circles
    for _ in range(200):
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                if dist < 0.2:
                    force = (0.2 - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
            # Boundary repulsion
            if x[i] < 0.15:
                fx[i] += (0.15 - x[i]) * 2
            if x[i] > 0.85:
                fx[i] -= (x[i] - 0.85) * 2
            if y[i] < 0.15:
                fy[i] += (0.15 - y[i]) * 2
            if y[i] > 0.85:
                fy[i] -= (y[i] - 0.85) * 2
        x += 0.01 * fx
        y += 0.01 * fy
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
    
    # Set radii based on minimum distance to neighbors
    r = np.zeros(N)
    for i in range(N):
        min_dist = min(x[i], 1 - x[i], y[i], 1 - y[i])
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                min_dist = min(min_dist, d)
        r[i] = min_dist * 0.45
    
    # Ensure circles fit in boundary
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = min(r[i], max_r * 0.95)
    
    return np.column_stack([x, y, r])

def _optimize_nearby(xyz, time_budget=15.0):
    """
    Optimize using nearby-pairs constraints, then polish with full pairwise.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Nearby-pairs optimization
    try:
        result = minimize(_objective_nearby, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0, 0.3),
                          options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _objective_nearby(best_params, 50.0, 0.3):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.6:
        return best_params.reshape(N, 3)
    
    # Phase 2: Polish with full pairwise constraints
    try:
        result = minimize(_objective_full, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(100.0,),
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.fun < _objective_full(best_params, 100.0):
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
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Force-directed
    initial_configs.append(_initial_config_force_directed())
    
    # 4. Random configurations
    np.random.seed(42)
    for trial in range(3):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 5. Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.22 * np.cos(angle)
        y = 0.5 + 0.22 * np.sin(angle)
        positions.append([x, y, 0.09])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.08])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.48 * np.cos(angle)
        y = 0.5 + 0.48 * np.sin(angle)
        positions.append([x, y, 0.07])
    
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        time_for_opt = min(15.0, remaining - 2.0)
        if time_for_opt < 2.0:
            break
        
        xyz_opt = _optimize_nearby(xyz0, time_budget=time_for_opt)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score and _validate(xyz_opt):
            best_score = score
            best_xyz = xyz_opt
    
    # Final high-precision polish on the best configuration
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 5.0:
        try:
            params = best_xyz.flatten()
            bounds = []
            for i in range(N):
                bounds.append((0, 1))
                bounds.append((0, 1))
                bounds.append((0, 0.5))
            
            result = minimize(_objective_full, params, method='L-BFGS-B',
                              bounds=bounds, args=(200.0,),
                              options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < _objective_full(params, 200.0):
                polished = result.x.reshape(N, 3)
                if _validate(polished) and np.sum(polished[:, 2]) > best_score:
                    best_score = np.sum(polished[:, 2])
                    best_xyz = polished
        except Exception:
            pass
    
    # Final check
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
