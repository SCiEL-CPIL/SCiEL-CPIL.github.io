"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 26

# EVOLVE-BLOCK-START

def _get_initial_configs():
    """Generate diverse initial configurations for circle packing."""
    configs = []
    
    # 1. Grid 5x5 + 1 center
    pos = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            pos.append([x, y])
    pos.append([0.5, 0.5])
    pos = np.array(pos[:N])
    radii = np.full(N, 0.09)
    configs.append((pos, radii))
    
    # 2. Hexagonal packing (approximate)
    pos = []
    # 6 rows of varying counts to get 26
    # Row 0: 5, Row 1: 4, Row 2: 5, Row 3: 4, Row 4: 5, Row 5: 3? 
    # Let's do a denser hex pattern
    # Spacing
    dx = 0.18
    dy = 0.155
    for r in range(7):
        for c in range(6):
            if len(pos) >= N:
                break
            x = (c + 0.5) * dx
            y = (r + 0.5) * dy
            if r % 2 == 1:
                x += dx / 2.0
            if 0 <= x <= 1 and 0 <= y <= 1:
                pos.append([x, y])
    pos = np.array(pos[:N])
    radii = np.full(N, 0.085)
    configs.append((pos, radii))
    
    # 3. Random with rejection sampling
    rng = np.random.RandomState(42)
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pos = []
        min_dist = 0.12
        attempts = 0
        while len(pos) < N and attempts < 2000:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            ok = True
            for px, py in pos:
                if (x - px)**2 + (y - py)**2 < min_dist**2:
                    ok = False
                    break
            if ok:
                pos.append([x, y])
            attempts += 1
        # If not enough, add random
        while len(pos) < N:
            pos.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        pos = np.array(pos)
        radii = np.full(N, 0.05)
        configs.append((pos, radii))
        
    return configs

def _compute_radii(pos, n):
    """
    Compute maximum radii for given positions.
    Uses iterative relaxation to handle dependencies.
    """
    radii = np.zeros(n)
    # Initialize with boundary limits
    for i in range(n):
        radii[i] = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
    
    # Iterate to resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            max_r = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_max_j = dist - radii[j]
                if r_max_j < max_r:
                    max_r = r_max_j
            if max_r < 0:
                max_r = 0.0
            if abs(max_r - radii[i]) > 1e-12:
                radii[i] = max_r
                changed = True
        if not changed:
            break
            
    return radii

def _penalty(pos, radii, n):
    """Calculate penalty for invalid packing."""
    penalty = 0.0
    # Boundary violation
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < 0: penalty += (x - r)**2
        if y - r < 0: penalty += (y - r)**2
        if x + r > 1: penalty += (x + r - 1)**2
        if y + r > 1: penalty += (y + r - 1)**2
        
    # Overlap violation
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    return penalty

def _objective(x, n):
    """
    Objective function for optimization.
    Minimizes negative sum of radii + penalty.
    x = [x1, y1, ..., xn, yn, r1, ..., rn]
    """
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Clip radii to be positive
    radii = np.maximum(radii, 0)
    
    pen = _penalty(pos, radii, n)
    sum_r = np.sum(radii)
    
    # We want to maximize sum_r, so minimize -sum_r + penalty
    return -sum_r + 100.0 * pen

def _local_search(pos, radii, n, deadline, rng):
    """
    Simple local search to polish the packing.
    """
    current_pos = pos.copy()
    current_radii = radii.copy()
    current_sum = np.sum(current_radii)
    
    step_size = 0.02
    
    for it in range(200):
        if time.time() > deadline:
            break
            
        i = rng.randint(n)
        orig_x, orig_y = current_pos[i]
        
        best_new_sum = current_sum
        best_new_pos = current_pos.copy()
        best_new_radii = current_radii.copy()
        
        # Try several random moves
        for _ in range(20):
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            new_x = np.clip(orig_x + dx, 0.0, 1.0)
            new_y = np.clip(orig_y + dy, 0.0, 1.0)
            
            trial_pos = current_pos.copy()
            trial_pos[i] = [new_x, new_y]
            
            trial_radii = _compute_radii(trial_pos, n)
            trial_sum = np.sum(trial_radii)
            
            if trial_sum > best_new_sum:
                best_new_sum = trial_sum
                best_new_pos = trial_pos
                best_new_radii = trial_radii
                
        if best_new_sum > current_sum + 1e-8:
            current_pos = best_new_pos
            current_radii = best_new_radii
            current_sum = best_new_sum
        else:
            step_size *= 0.95
            if step_size < 0.001:
                break
                
    return current_pos, current_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    configs = _get_initial_configs()
    
    rng = np.random.RandomState(123)
    
    for pos, radii in configs:
        if time.time() > deadline - 10:
            break
            
        try:
            # Optimization using scipy
            # Variables: positions (2*N) + radii (N)
            n = N
            x0 = np.zeros(3 * n)
            x0[:2*n] = pos.flatten()
            x0[2*n:] = radii
            
            # Use L-BFGS-B with bounds
            bounds = []
            for i in range(n):
                bounds.append((0.0, 1.0)) # x
                bounds.append((0.0, 1.0)) # y
            for i in range(n):
                bounds.append((0.0, 0.5)) # r (max possible is 0.5)
                
            # We optimize -sum(r) + penalty
            # Since radii are variables, we can't just fix them.
            # However, for a given position, optimal radius is determined.
            # So we can optimize positions only, and compute radii afterwards.
            # Let's try optimizing positions only first.
            
            def obj_pos(p_flat):
                p = p_flat.reshape(n, 2)
                r = _compute_radii(p, n)
                # Penalty for invalid? _compute_radii should ensure validity mostly, 
                # but might be slow or stuck. 
                # Let's add a small penalty if radii are 0 (stuck)
                return -np.sum(r)
                
            x0_pos = pos.flatten()
            bounds_pos = [(0.0, 1.0)] * (2 * n)
            
            res = minimize(obj_pos, x0_pos, method='L-BFGS-B', bounds=bounds_pos, options={'maxiter': 100, 'maxfun': 1000})
            
            opt_pos = res.x.reshape(n, 2)
            opt_radii = _compute_radii(opt_pos, n)
            
            # Local search polish
            opt_pos, opt_radii = _local_search(opt_pos, opt_radii, n, deadline - 5, rng)
            
            score = np.sum(opt_radii)
            
            if score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = opt_pos
                result[:, 2] = opt_radii
                best_result = result
                
        except Exception:
            continue
            
    if best_result is None:
        # Fallback
        result = np.zeros((N, 3))
        for i in range(N):
            result[i, 0] = (i % 5 + 0.5) / 5.0
            result[i, 1] = (i // 5 + 0.5) / 5.0
            result[i, 2] = 0.09
        return result
        
    return best_result

# EVOLVE-BLOCK-END
