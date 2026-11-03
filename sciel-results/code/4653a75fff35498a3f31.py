"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _penalty_objective(params, n, penalty_weight):
    """
    Objective function: minimize -sum(radii) + penalty_weight * violations.
    params: array of shape (3*n,) -> [x0, y0, r0, x1, y1, r1, ...]
    """
    # Unpack
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Clip radii to be non-negative for calculation, though bounds handle this
    radii = np.maximum(radii, 0.0)
    
    # Main objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Penalty terms
    violations = 0.0
    
    # 1. Boundary constraints: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    # Violation is max(0, r_i - bound_i)
    bounds = np.minimum(
        np.minimum(positions[:, 0], positions[:, 1]),
        np.minimum(1.0 - positions[:, 0], 1.0 - positions[:, 1])
    )
    # Ensure bounds are non-negative for penalty calc (if pos is outside, bound is neg, but pos should be bounded by optimizer)
    # Actually, if pos is outside [0,1], the circle is definitely invalid.
    # Let's penalize position being outside [0,1] as well if we don't bound it strictly in optimizer.
    # But we use L-BFGS-B with bounds, so positions are in [0,1].
    # Radii are in [0, 0.5] (max possible in unit square).
    
    # Calculate boundary violation: r_i > bound_i
    b_viol = np.maximum(0.0, radii - bounds)
    violations += np.sum(b_viol ** 2)
    
    # 2. Overlap constraints: dist_ij >= r_i + r_j
    # We can vectorize this for speed, but n=26 is small.
    # dist_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    # violation_ij = max(0, r_i + r_j - dist_ij)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                viol = min_dist - dist
                violations += viol ** 2
                
    return obj + penalty_weight * violations

def _is_valid(params, n, tol=1e-7):
    """Check if configuration is strictly valid."""
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Check radii non-negative
    if np.any(radii < -tol):
        return False
        
    # Check bounds
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1.0 + tol or y + r > 1.0 + tol:
            return False
            
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
                
    return True

def _repair(params, n):
    """
    Repair a potentially invalid configuration by reducing radii to satisfy constraints.
    This is a fallback to ensure validity if the optimizer leaves slight violations.
    """
    positions = params[:2*n].reshape(n, 2).copy()
    radii = params[2*n:].copy()
    
    # 1. Enforce boundary constraints strictly
    for i in range(n):
        x, y = positions[i]
        max_r = min(x, y, 1.0 - x, 1.0 - y)
        if radii[i] > max_r:
            radii[i] = max(0.0, max_r)
            
    # 2. Enforce non-overlap constraints strictly via iterative relaxation
    # We reduce radii of overlapping circles.
    # A simple strategy: if r_i + r_j > d_ij, reduce both by excess/2.
    # Iterate until no violations or max iterations.
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    excess = min_dist - dist
                    # Reduce both by half the excess
                    # Ensure we don't go below 0
                    red_i = excess / 2.0
                    red_j = excess / 2.0
                    
                    new_r_i = radii[i] - red_i
                    new_r_j = radii[j] - red_j
                    
                    if new_r_i < 0:
                        new_r_i = 0.0
                        # If one hits 0, the other can take the full reduction?
                        # Actually, if r_i becomes 0, then constraint is r_j <= dist.
                        # Let's just clip and let next iteration handle it if needed.
                        # But for strict validity, we need r_i + r_j <= dist.
                        # If new_r_i < 0, set to 0. Then check if new_r_j <= dist.
                        # If not, reduce new_r_j further.
                        if new_r_j > dist:
                            new_r_j = max(0.0, dist)
                            
                    if new_r_j < 0:
                        new_r_j = 0.0
                        if new_r_i > dist:
                            new_r_i = max(0.0, dist)
                            
                    if new_r_i != radii[i] or new_r_j != radii[j]:
                        radii[i] = new_r_i
                        radii[j] = new_r_j
                        changed = True
                        
        if not changed:
            break
            
    # Final check: ensure no negative radii
    radii = np.maximum(radii, 0.0)
    
    result = np.zeros(3*n)
    result[:2*n] = positions.flatten()
    result[2*n:] = radii
    return result

def _initial_grid(n):
    """Create a grid layout."""
    side = int(np.ceil(np.sqrt(n)))
    positions = np.zeros((n, 2))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (j + 0.5) / side
            y = (i + 0.5) / side
            positions[idx] = [x, y]
            idx += 1
    # Initial radius estimate: slightly less than half the cell width
    r0 = 0.5 / side * 0.8
    return positions, r0

def _initial_hexagonal(n):
    """Create a hexagonal layout."""
    positions = np.zeros((n, 2))
    # Approximate spacing for 26 circles in unit square
    # ~5x5 grid with offset
    spacing = 0.19
    row_spacing = spacing * np.sqrt(3) / 2
    idx = 0
    for row in range(8):
        for col in range(8):
            if idx >= n:
                break
            x = (col + 0.5) * spacing
            y = (row + 0.5) * row_spacing
            if row % 2 == 1:
                x += spacing / 2
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions[idx] = [x, y]
            idx += 1
    while idx < n:
        positions[idx] = [0.5, 0.5]
        idx += 1
    r0 = 0.08
    return positions, r0

def _initial_random(n, seed):
    """Create a random layout."""
    rng = np.random.RandomState(seed)
    positions = rng.uniform(0.1, 0.9, size=(n, 2))
    r0 = 0.05
    return positions, r0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Generate initial configurations
    configs = []
    pos_g, r_g = _initial_grid(N)
    configs.append((pos_g, r_g))
    
    pos_h, r_h = _initial_hexagonal(N)
    configs.append((pos_h, r_h))
    
    for seed in [1, 2, 3]:
        pos_r, r_r = _initial_random(N, seed)
        configs.append((pos_r, r_r))
    
    # Annealing schedule for penalty weight
    # Start low, increase to high
    penalty_schedule = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    for idx, (init_pos, init_r) in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
            
        # Initial params
        params = np.zeros(3 * N)
        params[:2 * N] = init_pos.flatten()
        params[2 * N:] = init_r
        
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        
        current_params = params.copy()
        
        # Run annealing
        for p_weight in penalty_schedule:
            if time.time() > deadline - 5.0:
                break
                
            try:
                result = minimize(
                    _penalty_objective,
                    current_params,
                    args=(N, p_weight),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                current_params = result.x
            except Exception:
                pass
                
        # Repair the final parameters to ensure validity
        try:
            repaired_params = _repair(current_params, N)
            
            # Validate
            if _is_valid(repaired_params, N):
                score = np.sum(repaired_params[2 * N:])
                if score > best_score:
                    best_score = score
                    best_result = repaired_params.reshape(N, 3)
        except Exception:
            continue
            
    if best_result is None:
        # Fallback: return the grid seed (which is valid)
        pos_g, r_g = _initial_grid(N)
        result = np.zeros((N, 3))
        result[:, :2] = pos_g
        result[:, 2] = r_g
        return result
        
    return best_result

# EVOLVE-BLOCK-END
