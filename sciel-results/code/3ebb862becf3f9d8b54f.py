import numpy as np
import random
import time

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    Uses Simulated Annealing with a focus on the worst-case distance.
    """
    
    # 1. Initialization
    # Use a simple random initialization within the square, avoiding boundary too closely initially
    # Or a grid initialization. Grid is often better to start.
    # Let's try a grid of 7x7 (49) + 1 center, or 5x10.
    # 50 points. 
    # A 7x7 grid has 49 points. The 50th can be placed in the center gap if we shift? 
    # Actually, 50 is not a perfect square. 
    # Let's use a random initialization but jittered around a grid to break symmetry.
    
    def initialize():
        # Create a 7x7 grid (49 points)
        points = []
        side = 7
        # Spacing to fit in [-0.4, 0.4] roughly
        spacing = 0.8 / (side - 1)
        for i in range(side):
            for j in range(side):
                x = -0.4 + i * spacing
                y = -0.4 + j * spacing
                # Add small random noise to break symmetry
                x += random.uniform(-0.01, 0.01)
                y += random.uniform(-0.01, 0.01)
                points.append([x, y])
        
        # We have 49 points. Need 1 more.
        # Place the 50th point at the center (0,0) with noise? 
        # The center of a 7x7 grid is occupied by the middle point (3,3).
        # So we need to displace one point or add a new one.
        # Better: Use a 5x10 grid? 50 points.
        # Let's restart with a 5x10 grid.
        
        points = []
        rows, cols = 5, 10
        # Fit in [-0.45, 0.45]
        # Width = 0.9, Height = 0.9
        # Spacing x: 0.9 / 9 = 0.1
        # Spacing y: 0.9 / 4 = 0.225
        # This is anisotropic. 
        # Let's just use random initialization in [-0.4, 0.4] for robustness.
        # Random initialization with SA often works if SA is strong enough.
        
        for _ in range(N):
            x = random.uniform(-0.45, 0.45)
            y = random.uniform(-0.45, 0.45)
            points.append([x, y])
            
        return np.array(points)

    points = initialize()
    
    # 2. Objective Function
    # We want to maximize r = min( min_{i<j} ||c_i - c_j||, min_i dist_to_boundary(c_i) )
    # Note: The task defines radius r such that circles are disjoint.
    # Distance between centers must be >= 2r.
    # Distance to boundary must be >= r.
    # So r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    
    def get_radius_and_worst(pts):
        # Calculate minimum distance between any pair
        # O(N^2) is fine for N=50
        min_dist_pair = float('inf')
        worst_i, worst_j = -1, -1
        
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist_pair:
                    min_dist_pair = d
                    worst_i, worst_j = i, j
                    
        # Calculate minimum distance to boundary
        min_dist_bound = float('inf')
        worst_wall_i, worst_wall_axis = -1, -1
        # Walls: x=-0.5, x=0.5, y=-0.5, y=0.5
        # dist to x=-0.5 is 0.5 + x
        # dist to x=0.5 is 0.5 - x
        # dist to y=-0.5 is 0.5 + y
        # dist to y=0.5 is 0.5 - y
        
        for i in range(N):
            d_left = 0.5 + pts[i][0]
            d_right = 0.5 - pts[i][0]
            d_down = 0.5 + pts[i][1]
            d_up = 0.5 - pts[i][1]
            
            min_d = min(d_left, d_right, d_down, d_up)
            if min_d < min_dist_bound:
                min_dist_bound = min_d
                worst_wall_i = i
                # Determine which wall
                if min_d == d_left: worst_wall_axis = 0 # Move +x
                elif min_d == d_right: worst_wall_axis = 1 # Move -x
                elif min_d == d_down: worst_wall_axis = 2 # Move +y
                else: worst_wall_axis = 3 # Move -y

        # The radius is limited by the smaller of these two (scaled appropriately)
        # r = min( min_dist_pair / 2, min_dist_bound )
        # To maximize r, we can just maximize the "effective" min distance.
        # Let's define the score as r.
        
        r = min(min_dist_pair / 2.0, min_dist_bound)
        
        # Determine which constraint is active for the "worst" move
        # If min_dist_pair/2 < min_dist_bound, the pair is the bottleneck.
        # Else the wall is the bottleneck.
        
        if min_dist_pair / 2.0 < min_dist_bound:
            active_type = 'pair'
            active_i = worst_i
            active_j = worst_j
            active_axis = -1
        else:
            active_type = 'wall'
            active_i = worst_wall_i
            active_j = -1
            active_axis = worst_wall_axis
            
        return r, active_type, active_i, active_j, active_axis

    # 3. Simulated Annealing Loop
    
    current_r, _, _, _, _ = get_radius_and_worst(points)
    best_r = current_r
    best_points = points.copy()
    
    # Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 100000
    step_size = 0.05
    
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Keep under 120s
    
    for iter in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Get current state
        current_r, active_type, active_i, active_j, active_axis = get_radius_and_worst(points)
        
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # Generate a move
        # Strategy: Move the "worst" point (or one of the worst pair) in a random direction
        # Or move a random point.
        # Moving the worst point is more directed.
        
        if active_type == 'pair':
            # Move active_i or active_j
            # Randomly choose which one to move
            if random.random() < 0.5:
                move_idx = active_i
            else:
                move_idx = active_j
        else:
            move_idx = active_i
            
        # Random direction
        angle = random.uniform(0, 2 * np.pi)
        # Step size can be scaled by T
        move_mag = step_size * random.uniform(0.5, 1.5)
        dx = move_mag * np.cos(angle)
        dy = move_mag * np.sin(angle)
        
        # Propose move
        old_x, old_y = points[move_idx]
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds? 
        # If it goes out of bounds, the radius becomes 0 or negative (invalid).
        # It's better to clip or reject.
        # Clipping might lead to points sticking to walls.
        # Let's just allow it and calculate the new radius.
        # If it goes out, dist to boundary is negative, so r is negative.
        # This will be rejected unless T is high enough.
        
        points[move_idx] = [new_x, new_y]
        
        # Calculate new radius
        new_r, _, _, _, _ = get_radius_and_worst(points)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            # Always accept improvement
            pass
        else:
            # Accept with probability exp(delta_r / T)
            # Note: delta_r is negative
            if random.random() < np.exp(delta_r / T):
                pass
            else:
                # Reject
                points[move_idx] = [old_x, old_y]
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck at low T, we might want to reheat or just stop.
            # Let's add a "shake" if we haven't improved in a while.
            
    # Final refinement: 
    # Sometimes SA ends with a point slightly outside or very close to boundary.
    # We can do a few iterations of "projection" or just return the best found.
    # The best_points are guaranteed to be the best found during the process.
    
    # Ensure best_points are within [-0.5, 0.5] 
    # If a point was outside, its radius contribution was negative, so it wouldn't have been the "best" 
    # unless all points were outside (unlikely).
    # However, if the best configuration had a point exactly on the boundary, it's valid.
    
    return best_points
