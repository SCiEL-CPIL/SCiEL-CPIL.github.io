"""Combined program: Multi-start force-directed relaxation with varied aspect ratios."""
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
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < (r[i] + r[j] - tol) ** 2:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors
            x_c = np.clip(x, 0, w)
            y_c = np.clip(y, 0, h)
            r_c = np.clip(r, 0, None)
            
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_c, y_c, r_c])
                return True
        return False

    def force_relax(w, h, x, y, r, max_iter=300, step_size=0.05, growth_rate=0.01):
        """
        Force-directed relaxation with radius growth.
        Returns (x, y, r, valid_flag)
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        n = len(x)
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Calculate forces
            fx.fill(0.0)
            fy.fill(0.0)
            
            # Circle-Circle Repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist > 1e-6:
                            overlap = min_dist - dist
                            force = overlap * 10.0
                            fx[i] += force * (dx / dist)
                            fy[i] += force * (dy / dist)
                            fx[j] -= force * (dx / dist)
                            fy[j] -= force * (dy / dist)
                        else:
                            # Jitter if exactly same position
                            fx[i] += random.uniform(-1, 1)
                            fy[i] += random.uniform(-1, 1)
                            fx[j] -= random.uniform(-1, 1)
                            fy[j] -= random.uniform(-1, 1)

            # Wall Repulsion
            for i in range(n):
                # Left
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                # Right
                if w - x[i] < r[i]:
                    fx[i] -= (r[i] - (w - x[i])) * 10.0
                # Bottom
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                # Top
                if h - y[i] < r[i]:
                    fy[i] -= (r[i] - (h - y[i])) * 10.0

            # 2. Update positions
            x += fx * step_size * 0.1
            y += fy * step_size * 0.1
            
            # 3. Clamp positions to valid range
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            
            # 4. Try to increase radii
            new_r = r * (1.0 + growth_rate)
            
            # Validate with new radii (cheap check for significant overlaps only)
            valid_growth = True
            for i in range(n):
                if new_r[i] > x[i] or new_r[i] > w - x[i] or new_r[i] > y[i] or new_r[i] > h - y[i]:
                    valid_growth = False
                    break
            if valid_growth:
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = new_r[i] + new_r[j]
                        if dist_sq < min_dist * min_dist:
                            valid_growth = False
                            break
                    if not valid_growth:
                        break
            
            if valid_growth:
                r = new_r
            else:
                # If not valid, keep same radii but maybe shrink slightly to allow movement
                # Actually, just keep same radii for now
                pass

            # 5. Occasionally try to optimize w
            # Calculate bounding box of centers
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            span_x = x_max - x_min + 2 * np.mean(r)
            span_y = y_max - y_min + 2 * np.mean(r)
            
            if span_x > 0 and span_y > 0:
                target_aspect = span_x / span_y
                # Target w such that w/h = target_aspect and w+h=2
                # w/(2-w) = A => w = 2A / (1+A)
                target_w = 2.0 * target_aspect / (1.0 + target_aspect)
                # Clamp target_w
                target_w = np.clip(target_w, 0.5, 1.5)
                
                # Move w towards target
                w = 0.9 * w + 0.1 * target_w
                h = 2.0 - w
                
                # Re-clamp positions to new bounds
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                
                # If w changed, radii might now be invalid, check
                if not validate(w, h, x, y, r, tol=1e-8):
                     # Reduce radii to fit
                     r = r * 0.95

        return x, y, r, w, h

    def create_grid_init(w, seed=42):
        h = 2.0 - w
        # Determine grid dimensions
        aspect = w / h
        if aspect >= 1.0:
            n_cols = int(np.round(np.sqrt(N * aspect)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / aspect)))
            n_cols = int(np.ceil(N / n_rows))
        
        # Adjust to ensure N cells
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
        max_r = min(cell_w, cell_h) * 0.45  # Leave some gap
        
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * cell_w
                y[idx] = (i + 0.5) * cell_h
                r[idx] = max_r
                idx += 1
        return x, y, r, w, h

    # Multi-start strategy with varied aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3]
    random.seed(42)
    np.random.seed(42)
    
    # Shuffle aspect ratios
    random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Initialize with grid
        x, y, r, w, h = create_grid_init(w)
        
        # Start with smaller radii to allow movement
        r = r * 0.5
        
        # Run force relaxation
        x, y, r, w, h = force_relax(w, h, x, y, r, max_iter=200, step_size=0.05, growth_rate=0.01)
        
        # Update best
        update_best(w, h, x, y, r)
        
        # If we have time, do a finer polish
        if time.time() < deadline - 5.0:
            x, y, r, w, h = force_relax(w, h, x, y, r, max_iter=100, step_size=0.01, growth_rate=0.001)
            update_best(w, h, x, y, r)

    # Fallback if best_result is still None (should not happen)
    if best_result is None:
        x, y, r, w, h = create_grid_init(1.0)
        best_result = np.column_stack([x, y, r])

    return best_result
# EVOLVE-BLOCK-END
