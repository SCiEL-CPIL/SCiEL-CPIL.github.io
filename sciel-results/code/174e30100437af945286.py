import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_result = None
    
    # Predefined aspect ratios to try
    # w is width, h = 2 - w
    # We want to cover a range of shapes.
    # Square is 1.0. 
    # Let's try a few.
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.9, 0.8, 0.7]
    
    # To maximize diversity, we shuffle the order of aspect ratios in each run?
    # No, let's be deterministic but cover the space well.
    # We will run the packing algorithm for each aspect ratio.
    
    def pack_for_aspect(w, h, seed_val):
        """
        Attempts to pack N circles in a w x h rectangle.
        Returns (sum_r, array_of_circles) or (-inf, None) if fails.
        """
        rng = np.random.default_rng(seed_val)
        
        # 1. Initialization: Hexagonal or Square Grid
        # Estimate optimal radius for grid
        # For N=21, a 5x5 grid is 25, so 4x5=20 or 5x4=20 or 3x7=21.
        # Let's try to create a grid that fits N circles.
        
        # Simple approach: Place circles in a grid pattern.
        # Determine grid dimensions
        # Try a few grid configurations
        best_local_score = -np.inf
        best_local_result = None
        
        # Grid shapes: (rows, cols)
        # We want rows*cols >= N
        # Let's try a few
        grid_shapes = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (2, 11), (11, 2)]
        
        for rows, cols in grid_shapes:
            if rows * cols < N:
                continue
                
            # Calculate grid spacing
            # To fit in w x h:
            # Spacing in x: w / (cols + 1) ? No, if radius is r, center must be in [r, w-r].
            # Let's assume uniform spacing.
            # x_i = (i + 0.5) * (w / cols)
            # y_j = (j + 0.5) * (h / rows)
            
            # Max possible radius for this grid is min(w/cols, h/rows) / 2 
            # But we can do better with hexagonal packing.
            # Let's start with a square grid and then optimize.
            
            # Initial positions
            xs = []
            ys = []
            for i in range(cols):
                for j in range(rows):
                    if len(xs) >= N:
                        break
                    x = (i + 0.5) * (w / cols)
                    y = (j + 0.5) * (h / rows)
                    xs.append(x)
                    ys.append(y)
            xs = np.array(xs[:N])
            ys = np.array(ys[:N])
            
            # Initial radius: small
            r_max_possible = min(w / cols, h / rows) / 2.0
            r = np.full(N, r_max_possible * 0.8)
            
            # Check initial validity
            # If initial setup is invalid (overlaps), we need to shrink or move.
            # With this grid, if r <= min(w/cols, h/rows)/2, they don't overlap.
            # So this is a valid start.
            
            # 2. Optimization Loop: Greedy Growth + Nudging
            # We will try to increase radii.
            # Strategy:
            # - Pick a random circle.
            # - Try to increase r by delta.
            # - If valid, accept.
            # - If invalid, try to nudge positions to resolve conflict.
            
            current_r = r.copy()
            current_x = xs.copy()
            current_y = ys.copy()
            
            # Annealing parameters
            delta_r = 0.01
            nudge_strength = 0.05
            iterations = 20000  # Cap iterations
            
            for it in range(iterations):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random circle
                idx = rng.integers(0, N)
                
                # Try to grow
                new_r = current_r[idx] + delta_r
                if new_r > min(w, h) / 2.0:
                    continue
                    
                current_r[idx] = new_r
                
                # Check validity
                valid = True
                conflict_idx = -1
                
                # Boundary check
                if current_x[idx] < current_r[idx] or current_x[idx] > w - current_r[idx] or \
                   current_y[idx] < current_r[idx] or current_y[idx] > h - current_r[idx]:
                    valid = False
                    # Boundary conflict: move center towards center of box
                    if current_x[idx] < current_r[idx]:
                        current_x[idx] = current_r[idx] + 1e-5
                    elif current_x[idx] > w - current_r[idx]:
                        current_x[idx] = w - current_r[idx] - 1e-5
                    if current_y[idx] < current_r[idx]:
                        current_y[idx] = current_r[idx] + 1e-5
                    elif current_y[idx] > h - current_r[idx]:
                        current_y[idx] = h - current_r[idx] - 1e-5
                    # Re-check after nudging? For simplicity, if boundary violated, we just clamp.
                    # But clamping might cause overlap.
                    # Let's just reject if boundary is violated after clamp? 
                    # Actually, let's just reject growth if it violates boundary.
                    valid = False
                    current_r[idx] -= delta_r # revert
                    continue

                # Overlap check
                for j in range(N):
                    if j == idx:
                        continue
                    dx = current_x[idx] - current_x[j]
                    dy = current_y[idx] - current_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < current_r[idx] + current_r[j] - 1e-6:
                        valid = False
                        conflict_idx = j
                        break
                
                if valid:
                    # Accept growth
                    pass
                else:
                    # Revert growth
                    current_r[idx] -= delta_r
                    
                    # If we hit a conflict, try to nudge the two circles apart
                    if conflict_idx != -1:
                        # Push apart
                        dx = current_x[idx] - current_x[conflict_idx]
                        dy = current_y[idx] - current_y[conflict_idx]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            # Random direction
                            angle = rng.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Push amount
                        push = nudge_strength * 0.1
                        current_x[idx] += dx * push
                        current_y[idx] += dy * push
                        current_x[conflict_idx] -= dx * push
                        current_y[conflict_idx] -= dy * push
                        
                        # Clamp to boundaries
                        current_x[idx] = np.clip(current_x[idx], current_r[idx], w - current_r[idx])
                        current_y[idx] = np.clip(current_y[idx], current_r[idx], h - current_r[idx])
                        current_x[conflict_idx] = np.clip(current_x[conflict_idx], current_r[conflict_idx], w - current_r[conflict_idx])
                        current_y[conflict_idx] = np.clip(current_y[conflict_idx], current_r[conflict_idx], h - current_r[conflict_idx])
                        
                        # Note: Nudging might create new overlaps or boundary violations.
                        # We don't check immediately. We rely on the next iteration's growth check 
                        # to detect invalidity. If the state is invalid, growth will fail, 
                        # and we might nudge again. 
                        # However, if the state is invalid, we should probably fix it.
                        # Let's add a "repair" step: if any overlap exists, nudge the worst pair.
                        
                    # Occasionally, do a global repair if stuck?
                    # For now, simple nudging is enough.
                    
                # Adaptive delta_r
                # If we succeed often, increase delta_r? 
                # If we fail often, decrease?
                # Simple heuristic: keep delta_r constant or slightly decay.
                if it % 1000 == 0:
                    delta_r = max(0.001, delta_r * 0.9)
                    nudge_strength = max(0.01, nudge_strength * 0.9)
            
            # Final Validity Check
            # Ensure the final configuration is strictly valid
            final_valid = True
            for i in range(N):
                if current_x[i] < current_r[i] - 1e-5 or current_x[i] > w - current_r[i] + 1e-5:
                    final_valid = False
                    break
                if current_y[i] < current_r[i] - 1e-5 or current_y[i] > h - current_r[i] + 1e-5:
                    final_valid = False
                    break
                
            if final_valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((current_x[i]-current_x[j])**2 + (current_y[i]-current_y[j])**2)
                        if dist < current_r[i] + current_r[j] - 1e-5:
                            final_valid = False
                            break
                    if not final_valid:
                        break
            
            if final_valid:
                score = np.sum(current_r)
                if score > best_local_score:
                    best_local_score = score
                    best_local_result = np.column_stack([current_x, current_y, current_r])
                    
        return best_local_score, best_local_result

    # Main Loop: Iterate over aspect ratios
    # We allocate time to each aspect ratio.
    num_aspects = len(aspect_ratios)
    time_per_aspect = (time_limit - 5.0) / num_aspects
    
    for k, w in enumerate(aspect_ratios):
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
            
        # Time check
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Run packing for this aspect ratio
        # Use a different seed for each aspect ratio to ensure diversity
        seed_val = 42 + k * 1000
        
        score, result = pack_for_aspect(w, h, seed_val)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
            
        # Optional: If we have time, run a second pass with different seeds?
        # For now, one pass per aspect ratio is a good structural change.
        
    if best_result is None:
        # Fallback: Simple grid
        # This should rarely happen
        w = 1.0
        h = 1.0
        xs = []
        ys = []
        r = []
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                if len(xs) >= N:
                    break
                x = (i + 0.5) * 0.2
                y = (j + 0.5) * 0.2
                xs.append(x)
                ys.append(y)
                r.append(0.1 - 1e-3)
        best_result = np.column_stack([xs, ys, r])
        
    return best_result
