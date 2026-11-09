import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Initial configuration: Grid
    side = 5  # 5x5 grid for 21 circles
    # Place in a 1x1 square initially
    # We want to maximize sum of radii.
    # Start with small radii.
    
    # Grid positions
    xs = []
    ys = []
    for i in range(N):
        row = i // side
        col = i % side
        # Center of cell
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        xs.append(x)
        ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)
    rs = np.full(N, 0.1)
    
    best_sum = np.sum(rs)
    best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
    
    # Pre-calculate pairs for overlap checks
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    step_size = 0.005
    push_iterations = 50
    shrink_factor = 0.99
    
    while time.time() - start_time < time_limit:
        # 1. Increase radii
        rs += step_size
        
        # 2. Resolve overlaps and boundary constraints
        for _ in range(push_iterations):
            # Check overlaps
            for i, j in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        # Random push
                        angle = np.random.rand() * 2 * np.pi
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Push apart
                    push = (min_dist - dist) * 0.5
                    xs[i] += dx * push
                    ys[i] += dy * push
                    xs[j] -= dx * push
                    ys[j] -= dy * push
            
            # Check bounding box
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            
            W = max_x - min_x
            H = max_y - min_y
            perimeter = 2 * (W + H)
            
            if perimeter > 4.0:
                # Scale down to fit
                scale = 4.0 / perimeter
                # Scale positions relative to center
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                
                xs = cx + (xs - cx) * scale
                ys = cy + (ys - cy) * scale
                rs = rs * scale
                
                # Reduce step size if we had to shrink
                step_size *= 0.95
            else:
                # If we have room, we can potentially increase step size slightly
                # But keep it stable
                step_size = min(step_size * 1.01, 0.01)
        
        # Jitter to escape local minima
        xs += np.random.normal(0, 0.01, N)
        ys += np.random.normal(0, 0.01, N)
        
        # Re-resolve overlaps after jitter
        for _ in range(push_iterations // 2):
            for i, j in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        angle = np.random.rand() * 2 * np.pi
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    push = (min_dist - dist) * 0.5
                    xs[i] += dx * push
                    ys[i] += dy * push
                    xs[j] -= dx * push
                    ys[j] -= dy * push
            
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            
            W = max_x - min_x
            H = max_y - min_y
            perimeter = 2 * (W + H)
            
            if perimeter > 4.0:
                scale = 4.0 / perimeter
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                
                xs = cx + (xs - cx) * scale
                ys = cy + (ys - cy) * scale
                rs = rs * scale
        
        current_sum = np.sum(rs)
        if current_sum > best_sum:
            best_sum = current_sum
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_rs = rs.copy()
            
    # Final check and formatting
    # Ensure the best configuration is valid
    min_x = np.min(best_xs - best_rs)
    max_x = np.max(best_xs + best_rs)
    min_y = np.min(best_ys - best_rs)
    max_y = np.max(best_ys + best_rs)
    W = max_x - min_x
    H = max_y - min_y
    perimeter = 2 * (W + H)
    
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        best_xs = cx + (best_xs - cx) * scale
        best_ys = cy + (best_ys - cy) * scale
        best_rs = best_rs * scale

    result = np.column_stack((best_xs, best_ys, best_rs))
    return result
