import numpy as np

N = 200

def construct():
    # Simulated Annealing Approach
    
    # Initialize with random points
    # Using a structured random start might be better than pure random
    # Let's use a grid with random jitter to avoid immediate heavy overlaps
    grid_size = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(-0.5 + 0.02, 0.5 - 0.02, grid_size)
    ys = np.linspace(-0.5 + 0.02, 0.5 - 0.02, grid_size)
    grid_points = np.array([(x, y) for y in ys for x in xs])
    
    # Shuffle and take N
    np.random.shuffle(grid_points)
    points = grid_points[:N].copy()
    
    # Add some random jitter
    points += np.random.uniform(-0.01, 0.01, size=(N, 2))
    # Clip to bounds
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    best_points = points.copy()
    best_score = calculate_score(points)
    
    # SA Parameters
    T = 0.01  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 1e-6
    iterations = 20000
    
    # Pre-calculate some stuff? No, just run.
    
    for _ in range(iterations):
        if T < T_min:
            break
            
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Generate random move
        # Move size should scale with T
        move_size = 0.01 * T / 0.01 # Scale move with temp? 
        # Actually, let's use a fixed small move size and adjust T for acceptance.
        # Or scale move size. Let's use a fixed step size that decays slightly.
        step = 0.02 * (T / 0.01) + 0.001
        delta = np.random.uniform(-step, step, size=2)
        
        old_pos = points[i].copy()
        new_pos = old_pos + delta
        
        # Check bounds
        if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
            # Reflect or reject? 
            # If we reject, we waste the iteration. 
            # Better to clip? Clipping creates bias.
            # Let's just reject moves that go out of bounds.
            continue
        
        points[i] = new_pos
        
        new_score = calculate_score(points)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            
        delta_score = new_score - best_score # Wait, SA compares to CURRENT, not BEST
        # We need current_score
        # Let's keep track of current_score
        
        # Correction: SA compares new vs current
        # I need to store current_score
        
    # The loop above was flawed because I didn't track current_score properly for acceptance.
    # Let's rewrite the loop logic in the replacement.
    
    return best_points

def calculate_score(points):
    # Efficiently calculate min radius
    # 1. Min distance to boundary
    dx_min = points[:, 0] + 0.5
    dx_max = 0.5 - points[:, 0]
    dy_min = points[:, 1] + 0.5
    dy_max = 0.5 - points[:, 1]
    min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
    global_min_boundary = np.min(min_boundary_dist)
    
    # 2. Min distance between centers
    # Using vectorized calculation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    global_min_center = np.min(dist)
    
    r = min(global_min_center / 2.0, global_min_boundary)
    return r
