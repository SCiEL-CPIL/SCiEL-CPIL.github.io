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

def _compute_all_pairs_penalty(x, y, r, beta=0.01, weight=100.0):
    """
    Compute smooth penalty for all pairs using log-sum-exp.
    This is O(N^2) but N=26 is small, and it avoids Delaunay discontinuities.
    """
    n = len(x)
    # Vectorized distance calculation
    # x, y, r are 1D arrays of length n
    # Create NxN distance matrix
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Required distance matrix
    req_dist = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Overlap = req_dist - dist
    # We only care about positive overlaps
    overlaps = req_dist - dist
    
    # Mask out diagonal (self-overlap) and negative overlaps (valid)
    np.fill_diagonal(overlaps, -np.inf)
    
    # We want to penalize positive overlaps.
    # Use log-sum-exp on positive parts.
    # To make it smooth and differentiable everywhere, we can use a soft-min or just penalize max(0, overlap).
    # The current code uses log-sum-exp on positive overlaps.
    # Let's stick to that but vectorize it.
    
    # Identify positive overlaps
    pos_overlaps = overlaps[overlaps > 0]
    
    if len(pos_overlaps) == 0:
        return 0.0
    
    # Log-sum-exp
    max_ov = np.max(pos_overlaps)
    if max_ov < 1e-10:
        return 0.0
        
    lse = beta * (np.log(np.sum(np.exp((pos_overlaps - max_ov) / beta))) + max_ov / beta)
    return weight * (lse ** 2)

def _initial_config_hex():
    """
    Generate an initial configuration using a hexagonal-like arrangement.
    We'll use a grid with slight perturbations.
    """
    # Use a 5x5 grid (25 circles) plus one more
    # Actually, let's use a smarter initial layout
    # Start with a 5x5 grid
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            # Offset odd rows for hexagonal packing
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    # Add the 26th circle - try to place it in a gap
    # For now, just add another at a random-ish position
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

def _optimize_from(xyz, time_budget=30.0):
    """
    Optimize from a given initial configuration.
    Returns the best valid configuration found.
    Uses all-pairs smooth log-sum-exp penalty for robustness.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    def _boundary_penalty(x, y, r):
        pen = 0.0
        # Vectorized boundary penalty
        # x - r < 0 => (x-r)^2
        # x + r > 1 => (x+r-1)^2
        # etc.
        pen += np.sum(np.maximum(0, r - x) ** 2)
        pen += np.sum(np.maximum(0, x + r - 1) ** 2)
        pen += np.sum(np.maximum(0, r - y) ** 2)
        pen += np.sum(np.maximum(0, y + r - 1) ** 2)
        pen += np.sum(np.maximum(0, -r) ** 2)
        return pen

    def obj(p, beta=0.01, weight=100.0):
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        
        b_pen = _boundary_penalty(x, y, r)
        o_pen = _compute_all_pairs_penalty(x, y, r, beta=beta, weight=weight)
        
        return -np.sum(r) + b_pen + o_pen
    
    # Pass 1: Moderate penalty, find a good basin
    try:
        result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.fun < obj(best_params):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.5:
        # Pass 2: Higher precision, tighter penalty
        def obj_polish(p):
            x = p[0::3]
            y = p[1::3]
            r = p[2::3]
            
            b_pen = _boundary_penalty(x, y, r)
            # Smaller beta for sharper penalty, higher weight
            o_pen = _compute_all_pairs_penalty(x, y, r, beta=0.005, weight=500.0)
            
            return -np.sum(r) + b_pen + o_pen
        
        try:
            result2 = minimize(obj_polish, best_params, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result2.fun < obj_polish(best_params):
                best_params = result2.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    # Final validation
    xyz_best = best_params.reshape(N, 3)
    
    # Repair step: if invalid, try to fix by reducing radii and clipping
    if not _validate(xyz_best):
        xyz_fixed = xyz_best.copy()
        for _ in range(50):
            if _validate(xyz_fixed):
                break
            # Reduce radii slightly
            xyz_fixed[:, 2] *= 0.98
            # Clip positions
            for i in range(N):
                r = xyz_fixed[i, 2]
                xyz_fixed[i, 0] = np.clip(xyz_fixed[i, 0], r, 1 - r)
                xyz_fixed[i, 1] = np.clip(xyz_fixed[i, 1], r, 1 - r)
        
        if _validate(xyz_fixed):
            xyz_best = xyz_fixed
        else:
            # If still invalid, fall back to a valid grid
            return _initial_config_grid()
            
    return xyz_best

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Simple grid
    initial_configs.append(_initial_config_grid())
    
    # 2. Hexagonal-like
    initial_configs.append(_initial_config_hex())
    
    # 3. Random configurations with larger radii to encourage better packing
    np.random.seed(42)
    for trial in range(5):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.08, 0.18, N)
        # Ensure inside bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # 4. Concentric rings pattern
    positions = []
    # Center circle
    positions.append([0.5, 0.5, 0.12])
    # Ring 1
    ring1_n = 8
    ring1_dist = 0.25
    for k in range(ring1_n):
        angle = 2 * np.pi * k / ring1_n
        x = 0.5 + ring1_dist * np.cos(angle)
        y = 0.5 + ring1_dist * np.sin(angle)
        positions.append([x, y, 0.10])
    # Ring 2
    ring2_n = 10
    ring2_dist = 0.45
    for k in range(ring2_n):
        angle = 2 * np.pi * k / ring2_n + np.pi / ring2_n
        x = 0.5 + ring2_dist * np.cos(angle)
        y = 0.5 + ring2_dist * np.sin(angle)
        positions.append([x, y, 0.08])
    # Ring 3 (fill remaining)
    ring3_n = N - len(positions)
    if ring3_n > 0:
        ring3_dist = 0.48
        for k in range(ring3_n):
            angle = 2 * np.pi * k / ring3_n
            x = 0.5 + ring3_dist * np.cos(angle)
            y = 0.5 + ring3_dist * np.sin(angle)
            positions.append([x, y, 0.05])
            
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Give more time to earlier configs, less to later ones
        if idx < 3:
            time_budget = min(30.0, remaining - 2.0)
        else:
            time_budget = min(20.0, remaining - 2.0)
        if time_budget < 2.0:
            break
        
        xyz_opt = _optimize_from(xyz0, time_budget=time_budget)
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final check
    if not _validate(best_xyz):
        # Try to fix
        best_xyz = best_xyz.copy()
        for _ in range(200):
            if _validate(best_xyz):
                break
            best_xyz[:, 2] *= 0.95
            for i in range(N):
                x, y, r = best_xyz[i]
                best_xyz[i, 0] = np.clip(x, r, 1 - r)
                best_xyz[i, 1] = np.clip(y, r, 1 - r)
    
    # If still invalid, use the simple grid
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
