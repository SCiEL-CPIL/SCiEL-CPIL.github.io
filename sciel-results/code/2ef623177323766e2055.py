"""Improved program using gradient-based relaxation and radius expansion."""
import numpy as np
import time

N = 21

def _pack_for_aspect_ratio(w_init, max_iters=2000, seed=42):
    """
    Packs N circles in a rectangle of width w_init and height 2-w_init.
    """
    np.random.seed(seed)
    h = 2.0 - w_init
    
    # Initialize in a grid
    cols = int(np.sqrt(N * w_init / h)) + 1
    rows = (N + cols - 1) // cols
    
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // cols
        col = i % cols
        positions[i, 0] = (col + 0.5 + np.random.uniform(-0.1, 0.1)) * w_init / cols
        positions[i, 1] = (row + 0.5 + np.random.uniform(-0.1, 0.1)) * h / rows
        
    positions[:, 0] = np.clip(positions[:, 0], 0, w_init)
    positions[:, 1] = np.clip(positions[:, 1], 0, h)
    
    min_spacing_x = w_init / cols
    min_spacing_y = h / rows
    initial_r = 0.4 * min(min_spacing_x, min_spacing_y)
    radii = np.full(N, initial_r)
    
    best_score = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    learning_rate = 0.5
    growth_rate = 0.001
    stagnation_counter = 0
    max_stagnation = 50
    
    for it in range(max_iters):
        forces = np.zeros((N, 2))
        max_violation = 0.0
        
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            
            if x < r:
                v = r - x
                forces[i, 0] += v
                if v > max_violation: max_violation = v
            if x > w_init - r:
                v = x - (w_init - r)
                forces[i, 0] -= v
                if v > max_violation: max_violation = v
            if y < r:
                v = r - y
                forces[i, 1] += v
                if v > max_violation: max_violation = v
            if y > h - r:
                v = y - (h - r)
                forces[i, 1] -= v
                if v > max_violation: max_violation = v
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                
                if d_sq < r_sum * r_sum:
                    d = np.sqrt(d_sq)
                    if d < 1e-8:
                        dx = np.random.normal(0, 1e-8)
                        dy = np.random.normal(0, 1e-8)
                        d = np.sqrt(dx*dx + dy*dy)
                    
                    violation = r_sum - d
                    if violation > max_violation:
                        max_violation = violation
                    
                    ux = dx / d
                    uy = dy / d
                    
                    forces[i, 0] += ux * violation
                    forces[i, 1] += uy * violation
                    forces[j, 0] -= ux * violation
                    forces[j, 1] -= uy * violation
        
        positions += learning_rate * forces
        positions[:, 0] = np.clip(positions[:, 0], 0, w_init)
        positions[:, 1] = np.clip(positions[:, 1], 0, h)
        
        if max_violation < 1e-5:
            radii += growth_rate
            stagnation_counter = 0
            
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_positions = positions.copy()
                best_radii = radii.copy()
        else:
            stagnation_counter += 1
            if stagnation_counter > max_stagnation:
                positions += np.random.normal(0, 0.01, size=(N, 2))
                positions[:, 0] = np.clip(positions[:, 0], 0, w_init)
                positions[:, 1] = np.clip(positions[:, 1], 0, h)
                stagnation_counter = 0

    return best_score, best_positions, best_radii, w_init

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_overall_score = -np.inf
    best_overall_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for w in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        for seed in [1, 2, 3]:
            if time.time() - start_time > time_limit:
                break
                
            score, positions, radii, w_used = _pack_for_aspect_ratio(w, max_iters=1500, seed=seed)
            
            h_used = 2.0 - w_used
            
            # Cleanup
            for i in range(N):
                x, y, r = positions[i, 0], positions[i, 1], radii[i]
                x = np.clip(x, 0, w_used)
                y = np.clip(y, 0, h_used)
                positions[i, 0] = x
                positions[i, 1] = y
                max_r = min(x, w_used - x, y, h_used - y)
                if r > max_r:
                    radii[i] = max_r - 1e-7
                    
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = radii[i] + radii[j]
                    if d < r_sum - 1e-9:
                        overlap = r_sum - d
                        shrink = overlap / 2.0 + 1e-8
                        radii[i] -= shrink
                        radii[j] -= shrink
                        if radii[i] < 0: radii[i] = 0
                        if radii[j] < 0: radii[j] = 0
                        
            final_score = np.sum(radii)
            
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_result = np.column_stack([positions, radii])

    if best_overall_result is None:
        cols = 5
        rows = 5
        cells = []
        for i in range(cols):
            for j in range(rows):
                if len(cells) >= N: break
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                r = 0.5 / cols - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])
        
    return best_overall_result

# EVOLVE-BLOCK-END
