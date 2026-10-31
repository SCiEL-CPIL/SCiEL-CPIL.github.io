"""Circle packing in unit square - maximize sum of radii for 26 circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _initial_positions(n, seed=42):
    """Generate initial positions using a grid + jitter."""
    rng = np.random.default_rng(seed)
    # Use a grid layout as starting point
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add small jitter
                x += rng.uniform(-0.05, 0.05)
                y += rng.uniform(-0.05, 0.05)
                positions.append([x, y])
    positions = np.array(positions[:n])
    # Clip to [0.01, 0.99]
    positions = np.clip(positions, 0.01, 0.99)
    return positions

def _objective(params, n):
    """
    Objective: minimize negative sum of radii + penalty for violations.
    params: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    pos = params.reshape(n, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]
    
    # Negative sum of radii (we minimize, so negate)
    obj = -np.sum(r)
    
    # Penalty for circles going outside [0,1]^2
    # For each circle: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_violations = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                          np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    obj += 1000 * np.sum(boundary_violations)
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            overlap = np.maximum(0, min_dist - dist)
            obj += 1000 * overlap
    
    # Penalty for negative radii
    neg_r = np.maximum(0, -r)
    obj += 1000 * np.sum(neg_r)
    
    return obj

def _construct_with_optimization(n, time_limit=100, seed=42):
    """
    Use gradient-based optimization to pack n circles.
    """
    start_time = time.time()
    
    best_params = None
    best_obj = np.inf
    best_value = 0.0
    
    # Try multiple random restarts
    max_restarts = 5
    rng = np.random.default_rng(seed)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Generate initial positions
        pos = _initial_positions(n, seed=seed + restart * 100)
        
        # Initial radii: small
        r_init = np.full(n, 0.02)
        
        # Build initial parameter vector
        params = np.zeros(n * 3)
        params[0::3] = pos[:, 0]
        params[1::3] = pos[:, 1]
        params[2::3] = r_init
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.5))  # r
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                params,
                args=(n,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < best_obj:
                best_params = result.x
                best_obj = result.fun
                
                # Extract value
                pos_res = result.x.reshape(n, 3)
                r = pos_res[:, 2]
                # Check validity
                valid = True
                for i in range(n):
                    if r[i] < 0 or pos_res[i, 0] < r[i] or pos_res[i, 0] > 1 - r[i] or \
                       pos_res[i, 1] < r[i] or pos_res[i, 1] > 1 - r[i]:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = pos_res[i, 0] - pos_res[j, 0]
                            dy = pos_res[i, 1] - pos_res[j, 1]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
        except Exception:
            pass
    
    # If we have a valid result, return it
    if best_params is not None and best_value > 0:
        pos_res = best_params.reshape(n, 3)
        # Final cleanup: clip radii to ensure validity
        x = pos_res[:, 0]
        y = pos_res[:, 1]
        r = pos_res[:, 2]
        
        # Ensure circles stay within bounds
        for i in range(n):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = min(r[i], max_r * 0.999)
        
        # Final overlap check and fix
        for _ in range(10):
            fixed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-8:
                        # Reduce the smaller radius
                        if r[i] < r[j]:
                            r[i] = max(0, (dist - r[j]) * 0.999)
                        else:
                            r[j] = max(0, (dist - r[i]) * 0.999)
                        fixed = True
            if not fixed:
                break
        
        result = np.column_stack([x, y, r])
        return result
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = min(x, 1 - x, y, 1 - y) * 0.9
                cells.append([x, y, r])
    return np.array(cells[:n])

def construct():
    """
    Construct n disjoint circles inside the unit square maximizing sum of radii.
    Returns array of shape (n, 3): rows (x, y, radius).
    """
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s
    
    try:
        result = _construct_with_optimization(N, time_limit=time_limit)
    except Exception:
        # Fallback grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = min(x, 1 - x, y, 1 - y) * 0.9
                    cells.append([x, y, r])
        result = np.array(cells[:N])
    
    # Final validation and cleanup
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    
    # Clip positions and radii
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    r = np.clip(r, 0, 0.5)
    
    # Ensure circles are within bounds
    for i in range(N):
        max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = min(r[i], max_r * 0.9999)
    
    # Fix overlaps iteratively
    for _ in range(20):
        fixed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-9:
                    # Reduce radii proportionally
                    scale = dist / min_dist * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    fixed = True
        if not fixed:
            break
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
