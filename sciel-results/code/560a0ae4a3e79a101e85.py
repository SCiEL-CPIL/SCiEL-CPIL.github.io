"""Combined program: Physics-based circle packing with dynamic aspect ratio optimization."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    best_result = None
    best_score = -np.inf

    def validate(w, h, x, y, r, tol=1e-9):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < 0):
            return False
        # Check bounds
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        # Vectorized check for efficiency
        # Create distance matrix
        # To save memory and time for N=21, simple loops are fine, but let's try to be efficient
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors to ensure strict validity for evaluator
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            
            # Final strict check after clipping
            if validate(w, h, x_c, y_c, r_c, tol=1e-10):
                score = np.sum(r_c)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x_c, y_c, r_c])
                    return True
        return False

    def create_grid_init(w, seed=42):
        h = 2.0 - w
        aspect = w / h
        if aspect >= 1.0:
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        
        # Ensure we have enough cells
        while n_cols * n_rows < N:
            if aspect >= 1.0:
                n_cols += 1
            else:
                n_rows += 1
            
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / n_cols
        cell_h = h / n_rows
        max_r = min(cell_w, cell_h) * 0.45
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        return x, y, r

    def optimize_packing(w, h, x, y, r, max_iter=2000, time_limit=10.0):
        """
        Physics-based optimization.
        Forces:
        1. Circle-Circle repulsion (overlap).
        2. Wall repulsion (penetration).
        3. Centering force (prevents drift).
        4. Aspect ratio adjustment based on pressure.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # Cooling schedule
        step_size = 0.05
        decay = 0.995
        growth_rate = 1.005 # Try to grow radii by 0.5% each successful step
        
        start_opt_time = time.time()
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if time.time() - start_opt_time > time_limit:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # 1. Circle-Circle Forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist > 1e-8:
                            overlap = min_dist - dist
                            # Force is proportional to overlap
                            force = overlap * 50.0
                            fx[i] += force * (dx / dist)
                            fy[i] += force * (dy / dist)
                            fx[j] -= force * (dx / dist)
                            fy[j] -= force * (dy / dist)
                        else:
                            # Jitter
                            fx[i] += random.uniform(-1, 1)
                            fy[i] += random.uniform(-1, 1)
                            fx[j] -= random.uniform(-1, 1)
                            fy[j] -= random.uniform(-1, 1)

            # 2. Wall Forces
            # We treat walls as infinitely stiff. If a circle penetrates, push it back.
            for i in range(N):
                # Left
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 100.0
                # Right
                if w - x[i] < r[i]:
                    fx[i] -= (r[i] - (w - x[i])) * 100.0
                # Bottom
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 100.0
                # Top
                if h - y[i] < r[i]:
                    fy[i] -= (r[i] - (h - y[i])) * 100.0

            # 3. Centering Force (Weak)
            # Keeps the packing centered in the box
            cx = np.mean(x)
            cy = np.mean(y)
            fx += (w / 2.0 - cx) * 0.1
            fy += (h / 2.0 - cy) * 0.1

            # Update Positions
            x += fx * step_size
            y += fy * step_size
            
            # Clamp to box (Hard constraint)
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)

            # 4. Aspect Ratio Optimization
            # Calculate "pressure" or bounding box tightness
            # If circles are hitting left/right walls more than top/bottom, expand w
            # Simple heuristic: Compare span of x vs span of y relative to available space
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            # Effective width used by circles (including radius)
            # A more robust metric: How much space is left?
            space_x = w - (x_max - x_min) - 2*np.mean(r) # Rough approx
            space_y = h - (y_max - y_min) - 2*np.mean(r)
            
            # If space_x > space_y, we should shrink w and expand h? 
            # No, if space_x > space_y, it means we have more slack in x. 
            # To maximize density, we want to fill the slack. 
            # But we are constrained by perimeter. 
            # Let's use the ratio of extents.
            extent_x = x_max - x_min + 2*np.mean(r)
            extent_y = y_max - y_min + 2*np.mean(r)
            
            if extent_x > 0 and extent_y > 0:
                target_aspect = extent_x / extent_y
                # Map aspect to w
                # w / (2-w) = target_aspect => w = 2 * target_aspect / (1 + target_aspect)
                target_w = 2.0 * target_aspect / (1.0 + target_aspect)
                # Smoothly move towards target
                w = 0.9 * w + 0.1 * target_w
                h = 2.0 - w
                
                # Re-clamp positions to new bounds
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)

            # 5. Radius Growth
            # Try to grow radii
            new_r = r * growth_rate
            
            # Check if new radii are valid
            if validate(w, h, x, y, new_r, tol=1e-9):
                r = new_r
            else:
                # If growth fails, maybe shrink slightly to allow movement?
                # Or just keep same radii. 
                # Let's keep same radii to avoid oscillation, but reduce step size
                pass

            # Cool down
            step_size *= decay
            # Occasionally reheat? No, keep it simple.

            # Update best periodically
            if it % 100 == 0:
                update_best(w, h, x, y, r)

        # Final update
        update_best(w, h, x, y, r)
        
        return x, y, r, w, h

    # Multi-start strategy
    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15]
    random.seed(42)
    np.random.seed(42)
    
    # Shuffle to vary order
    random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Initialize with grid
        x, y, r = create_grid_init(w)
        
        # Start with smaller radii to allow movement
        r = r * 0.5
        
        # Run optimization
        x, y, r, w, h = optimize_packing(w, h, x, y, r, max_iter=1000, time_limit=10.0)
        
        # Polish phase: smaller steps, more iterations
        if time.time() < deadline - 10.0:
             x, y, r, w, h = optimize_packing(w, h, x, y, r, max_iter=2000, time_limit=5.0)

    # Fallback
    if best_result is None:
        x, y, r = create_grid_init(1.0)
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
