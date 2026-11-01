import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 0.0
    best_config = None
    
    # Try different aspect ratios
    # w + h = 2, so h = 2 - w
    # w in [0.5, 1.5] roughly
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for w in aspect_ratios:
        h = 2.0 - w
        if h <= 0:
            continue
            
        # Initialize positions in a grid
        # Estimate grid size
        n_x = int(np.round(np.sqrt(N * w / h)))
        n_y = int(np.round(N / n_x))
        if n_y * n_x < N:
            n_y += 1
            
        # Create grid points
        x_coords = np.linspace(0.5/n_x, 1 - 0.5/n_x, n_x)
        y_coords = np.linspace(0.5/n_y, 1 - 0.5/n_y, n_y)
        
        points = []
        for i in range(n_x):
            for j in range(n_y):
                if len(points) < N:
                    points.append([x_coords[i], y_coords[j]])
        
        points = np.array(points)
        
        # Scale to w, h
        points[:, 0] *= w
        points[:, 1] *= h
        
        # Local search to maximize min distance
        # We want to maximize d such that dist(p_i, p_j) >= 2d and dist(p_i, wall) >= d
        # So we maximize min( min_pair_dist, min_wall_dist )
        
        current_min_dist = 0.0
        for _ in range(2000): # Iterations per aspect ratio
            if time.time() > deadline:
                break
                
            # Calculate current min distance
            # Pairwise distances
            dists = []
            for i in range(N):
                for j in range(i+1, N):
                    dists.append(np.linalg.norm(points[i] - points[j]))
                # Wall distances
                dists.append(points[i][0])
                dists.append(w - points[i][0])
                dists.append(points[i][1])
                dists.append(h - points[i][1])
            
            min_dist = min(dists)
            
            if min_dist > current_min_dist:
                current_min_dist = min_dist
                
            # Move the point that is most constrained? Or random point?
            # Let's try moving a random point slightly
            idx = np.random.randint(N)
            step = 0.01 * (w + h) # Relative step
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            
            new_p = points[idx].copy()
            new_p[0] += dx
            new_p[1] += dy
            
            # Check if new point is within bounds (allowing some overlap with wall for now, 
            # but we will penalize via min_dist)
            # Actually, if it goes out of bounds, min_dist becomes negative.
            # We should constrain it to stay in bounds to be efficient.
            new_p[0] = np.clip(new_p[0], 0, w)
            new_p[1] = np.clip(new_p[1], 0, h)
            
            # Calculate new min dist with new point
            # Only need to check pairs involving idx
            new_dists = []
            for j in range(N):
                if j == idx:
                    continue
                new_dists.append(np.linalg.norm(new_p - points[j]))
            new_dists.append(new_p[0])
            new_dists.append(w - new_p[0])
            new_dists.append(new_p[1])
            new_dists.append(h - new_p[1])
            
            new_min_local = min(new_dists)
            
            # The global min is min(old_min_excluding_idx, new_min_local)
            # We don't have old_min_excluding_idx easily without recalculating.
            # For simplicity and speed, we just accept if new_min_local is not terrible, 
            # and rely on the global check in the next iteration's start.
            # Better: accept if new_min_local >= current_min_dist * 0.9 (allow slight decrease to escape local minima?)
            # Or just accept if new_min_local > min_dist (strict improvement).
            
            # Let's use a simple acceptance: accept if new_min_local > min_dist
            # This might get stuck. Let's add some noise.
            if new_min_local > min_dist or np.random.rand() < 0.1:
                 # Even if it decreases slightly, we might escape. 
                 # But we must not decrease too much.
                 if new_min_local > min_dist * 0.8:
                     points[idx] = new_p
        
        # After local search, calculate final score
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                dists.append(np.linalg.norm(points[i] - points[j]))
            dists.append(points[i][0])
            dists.append(w - points[i][0])
            dists.append(points[i][1])
            dists.append(h - points[i][1])
        
        min_d = min(dists)
        r = min_d / 2.0
        score = N * r
        
        if score > best_score:
            best_score = score
            best_config = points.copy()
            best_w = w
            best_h = h

    if best_config is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Scale best_config to perimeter 4
    # Current best_config is in w x h where w+h=2.
    # Perimeter is 2(w+h) = 4. So it's already scaled correctly for perimeter 4.
    # Wait, the points are in [0, w] x [0, h].
    # The radius r is min_d / 2.
    # We need to return (x, y, r) for each circle.
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_config[i][0]
        result[i][1] = best_config[i][1]
        result[i][2] = best_score / N # All radii equal
        
    return result
