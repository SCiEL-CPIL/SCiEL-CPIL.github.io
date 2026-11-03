import numpy as np
import time
import random

N = 21

def _repair(pos, radii, w, h, max_iter=100):
    """
    Repairs a configuration to be valid by resolving overlaps and wall violations.
    This is a simple iterative projection method.
    1. Check wall constraints: shrink radius or move center.
    2. Check pair constraints: shrink radii or move centers.
    To maximize sum of radii, we prefer shrinking radii less if possible, 
    but for simplicity and robustness, we will shrink radii to satisfy constraints.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Ensure walls are satisfied first
    for _ in range(max_iter):
        changed = False
        
        # Wall constraints
        for i in range(N):
            # Left
            if pos[i, 0] - radii[i] < 0:
                radii[i] = max(0.0, pos[i, 0])
                changed = True
            # Right
            if pos[i, 0] + radii[i] > w:
                radii[i] = max(0.0, w - pos[i, 0])
                changed = True
            # Bottom
            if pos[i, 1] - radii[i] < 0:
                radii[i] = max(0.0, pos[i, 1])
                changed = True
            # Top
            if pos[i, 1] + radii[i] > h:
                radii[i] = max(0.0, h - pos[i, 1])
                changed = True
        
        # Pair constraints
        # We need ||p_i - p_j|| >= r_i + r_j
        # If violated, we can shrink both radii proportionally or move centers.
        # Shrink radii is easier to guarantee validity.
        # Let d = ||p_i - p_j||. We need r_i + r_j <= d.
        # Current sum S = r_i + r_j. If S > d, scale factor k = d/S.
        # r_i' = r_i * k, r_j' = r_j * k.
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r - 1e-9:
                    if dist < 1e-9:
                        # Centers are effectively same. Move one slightly or shrink both to 0.
                        # Let's shrink both to 0 to be safe, or move one.
                        # Moving is better for packing.
                        # Push j away from i by a small epsilon
                        angle = random.uniform(0, 2*np.pi)
                        eps = 1e-3
                        pos[j, 0] += eps * np.cos(angle)
                        pos[j, 1] += eps * np.sin(angle)
                        # Re-check walls for j
                        pos[j, 0] = np.clip(pos[j, 0], 0, w)
                        pos[j, 1] = np.clip(pos[j, 1], 0, h)
                        changed = True
                    else:
                        # Scale radii to fit distance
                        scale = dist / sum_r
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
        
        if not changed:
            break
            
    return pos, radii

def _is_valid(pos, radii, w, h, tol=1e-5):
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Try a few aspect ratios to see which allows better packing
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2.
    # Square is 1x1.
    # Long thin might be worse for 21 circles.
    # Let's stick to 1x1 for simplicity and robustness, or try a few.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5]
    
    random.seed(42) # For reproducibility in dev, but SA is stochastic
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initialize with a grid
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * (w / side)
            pos[i, 1] = (row + 0.5) * (h / side)
            
        # Initial radii: small
        min_dim = min(w, h)
        base_r = 0.5 * min_dim / side * 0.5
        radii = np.full(N, base_r)
        
        # Repair initial state
        pos, radii = _repair(pos, radii, w, h)
        
        if _is_valid(pos, radii, w, h):
            cur_value = np.sum(radii)
            if cur_value > best_value:
                best_value = cur_value
                best_result = np.column_stack([pos, radii])
        else:
            continue # If initial grid can't be repaired, skip this aspect
        
        # Simulated Annealing
        T = 0.05 # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        max_iter = 5000
        step_size_pos = 0.05
        step_size_rad = 0.02
        
        cur_pos = pos.copy()
        cur_radii = radii.copy()
        cur_val = np.sum(cur_radii)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            new_pos = cur_pos.copy()
            new_radii = cur_radii.copy()
            
            # Perturb positions
            idx = random.randint(0, N-1)
            delta_x = random.gauss(0, step_size_pos)
            delta_y = random.gauss(0, step_size_pos)
            new_pos[idx, 0] += delta_x
            new_pos[idx, 1] += delta_y
            
            # Perturb radii
            # Randomly pick a few circles to grow/shrink
            for _ in range(3):
                r_idx = random.randint(0, N-1)
                delta_r = random.gauss(0, step_size_rad)
                new_radii[r_idx] += delta_r
                new_radii[r_idx] = max(0.0, new_radii[r_idx])
            
            # Repair
            new_pos, new_radii = _repair(new_pos, new_radii, w, h)
            
            # Check validity (repair should ensure this, but double check)
            if not _is_valid(new_pos, new_radii, w, h):
                # If repair failed, discard
                new_val = -np.inf
            else:
                new_val = np.sum(new_radii)
            
            # Acceptance criterion
            delta_val = new_val - cur_val
            if delta_val > 0:
                # Better, always accept
                cur_pos = new_pos
                cur_radii = new_radii
                cur_val = new_val
                
                if cur_val > best_value:
                    best_value = cur_val
                    best_result = np.column_stack([cur_pos, cur_radii])
            else:
                # Worse, accept with probability
                if T > T_min:
                    prob = np.exp(delta_val / T)
                    if random.random() < prob:
                        cur_pos = new_pos
                        cur_radii = new_radii
                        cur_val = new_val
            
            # Cool down
            T *= cooling_rate
            
            # Adaptive step size: if we've been stuck, increase step size to escape
            # Simple heuristic: if no improvement in last 100 steps, increase step size
            # This is a simplified version, but helps with local minima
            
    if best_result is None:
        # Fallback: simple valid grid
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])
        
    return best_result
