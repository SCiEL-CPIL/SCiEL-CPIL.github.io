"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for overhead
    
    best_sum = -1
    best_result = None
    
    # Try a few aspect ratios. w+h=2.
    # Square (1x1) is a strong candidate.
    # Also try 1.5 x 0.5 and 0.5 x 1.5 (long strips)
    aspect_ratios = [1.0, 1.5, 0.5, 1.2, 0.8]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize with a grid pattern
        # Estimate rows and cols based on aspect ratio
        # N=21. 
        # If ar=1, 5x5 grid is 25 cells.
        # If ar=1.5, 6x4 grid is 24 cells.
        
        # Simple heuristic for grid dimensions
        # Try to make rows * cols >= N and aspect of grid close to ar
        best_rows = 1
        best_cols = N
        min_diff = float('inf')
        for r in range(1, N+1):
            for c in range(1, N+1):
                if r * c >= N:
                    grid_ar = c / r
                    diff = abs(grid_ar - ar)
                    if diff < min_diff:
                        min_diff = diff
                        best_rows = r
                        best_cols = c
        
        rows = best_rows
        cols = best_cols
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                positions.append([x, y])
        
        positions = np.array(positions[:N])
        
        # Initial radii: small enough to not overlap
        # Max radius is limited by spacing
        dx = w / cols
        dy = h / rows
        min_spacing = min(dx, dy)
        max_r = min_spacing * 0.45
        radii = np.full(N, max_r)
        
        # State vector: [x1, y1, r1, x2, y2, r2, ...]
        state = np.zeros(3 * N)
        for i in range(N):
            state[3*i] = positions[i][0]
            state[3*i+1] = positions[i][1]
            state[3*i+2] = radii[i]
        
        # Simulated Annealing
        T = 0.1
        T_min = 1e-6
        cooling_rate = 0.9995
        steps = 50000
        step_size = 0.05
        
        current_state = state.copy()
        current_energy = calc_energy(current_state, w, h)
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random circle and a random parameter (x, y, or r)
            idx = np.random.randint(N)
            param = np.random.randint(3) # 0:x, 1:y, 2:r
            
            # Generate new state
            new_state = current_state.copy()
            new_val = current_state[3*idx + param] + np.random.normal(0, step_size)
            
            # Constraints on new value
            if param == 0: # x
                new_val = np.clip(new_val, 0.001, w - 0.001)
            elif param == 1: # y
                new_val = np.clip(new_val, 0.001, h - 0.001)
            else: # r
                new_val = np.clip(new_val, 0.001, min(w, h)/2)
            
            new_state[3*idx + param] = new_val
            
            new_energy = calc_energy(new_state, w, h)
            
            delta = new_energy - current_energy
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_state = new_state
                current_energy = new_energy
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Extract best result from this SA run
        xs = current_state[0::3]
        ys = current_state[1::3]
        rs = current_state[2::3]
        
        # Post-processing: Ensure validity by shrinking radii if needed
        # We want to maximize sum(rs) while keeping valid.
        # If there are overlaps or boundary violations, we must shrink.
        # A simple approach: iteratively shrink all radii by a small factor until valid.
        # Or, just check validity. If invalid, try to repair.
        
        # Let's try to repair by scaling radii down
        scale = 1.0
        max_iter = 100
        for _ in range(max_iter):
            valid, max_violation = check_validity(xs, ys, rs, w, h)
            if valid:
                break
            # Shrink radii by 1%
            rs = rs * 0.99
            scale *= 0.99
        
        if check_validity(xs, ys, rs, w, h)[0]:
            s = np.sum(rs)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack([xs, ys, rs])
    
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

def calc_energy(state, w, h):
    """
    Calculate energy (cost) for a given state.
    Lower is better.
    Energy = Overlap Penalty + Boundary Penalty - Lambda * Sum(Radii)
    """
    N = len(state) // 3
    xs = state[0::3]
    ys = state[1::3]
    rs = state[2::3]
    
    energy = 0.0
    
    # Boundary Penalty
    for i in range(N):
        if xs[i] < rs[i]:
            energy += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            energy += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            energy += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            energy += (ys[i] - (h - rs[i])) ** 2
    
    # Overlap Penalty
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                energy += (min_dist - dist) ** 2
    
    # Maximize sum of radii
    energy -= 1.0 * np.sum(rs)
    
    return energy

def check_validity(xs, ys, rs, w, h, tol=1e-6):
    """
    Check if the configuration is valid.
    Returns (is_valid, max_violation)
    """
    N = len(xs)
    max_violation = 0.0
    is_valid = True
    
    for i in range(N):
        if rs[i] <= 0:
            return False, 1.0
        if xs[i] < rs[i] - tol:
            max_violation = max(max_violation, rs[i] - xs[i])
            is_valid = False
        if xs[i] > w - rs[i] + tol:
            max_violation = max(max_violation, xs[i] - (w - rs[i]))
            is_valid = False
        if ys[i] < rs[i] - tol:
            max_violation = max(max_violation, rs[i] - ys[i])
            is_valid = False
        if ys[i] > h - rs[i] + tol:
            max_violation = max(max_violation, ys[i] - (h - rs[i]))
            is_valid = False
    
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            min_dist = rs[i] + rs[j]
            if dist < min_dist - tol:
                max_violation = max(max_violation, min_dist - dist)
                is_valid = False
                
    return is_valid, max_violation
# EVOLVE-BLOCK-END
