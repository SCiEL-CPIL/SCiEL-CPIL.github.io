import numpy as np

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing to maximize the minimum distance between circles
    and between circles and the square boundary.
    """
    
    def evaluate(points):
        """
        Returns the minimum radius r for the given configuration.
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        if points is None or len(points) != N:
            return -np.inf
        
        # Check if any point is outside (should not happen if we constrain moves, but good for safety)
        # The problem says "inside the square [-1/2, 1/2]^2".
        # If a point is outside, the "distance to boundary" is negative, so r becomes negative.
        
        # 1. Distance to walls
        # points shape (N, 2)
        # x in [-0.5, 0.5], y in [-0.5, 0.5]
        # dist to x=0.5 is 0.5 - x
        # dist to x=-0.5 is 0.5 + x
        # dist to y=0.5 is 0.5 - y
        # dist to y=-0.5 is 0.5 + y
        
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # 2. Distance between pairs
        # Calculate all pairwise distances
        # N=200, so 200x200 matrix is small.
        # dist[i, j] = ||p_i - p_j||
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def improve(points, temp):
        """
        Perform a single step of optimization.
        Strategy:
        1. Find the bottleneck (min wall or min pair).
        2. Move the involved points to increase the distance.
        3. Add random noise scaled by temp.
        """
        p = points.copy()
        
        # Calculate current min wall and min pair
        wall_dists = np.minimum.reduce([
            0.5 - p[:, 0],
            0.5 + p[:, 0],
            0.5 - p[:, 1],
            0.5 + p[:, 1]
        ])
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists[min_wall_idx]
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        min_pair_val = dists[i, j]
        
        current_r = min(min_wall_val, min_pair_val / 2.0)
        
        # Determine which constraint is active
        # We want to increase the minimum.
        # If wall is the bottleneck, move that point inward.
        # If pair is the bottleneck, move them apart.
        
        move_magnitude = 0.01 # Base step size
        
        if min_wall_val < min_pair_val / 2.0:
            # Wall is bottleneck
            idx = min_wall_idx
            # Determine direction to move: towards center
            # But we need to know which wall.
            x, y = p[idx]
            # Check which wall is closest
            dx_pos = 0.5 - x
            dx_neg = 0.5 + x
            dy_pos = 0.5 - y
            dy_neg = 0.5 + y
            
            moves = []
            if dx_pos == min(dx_pos, dx_neg, dy_pos, dy_neg):
                moves.append((-1, 0)) # Move left (towards -x)
            elif dx_neg == min(dx_pos, dx_neg, dy_pos, dy_neg):
                moves.append((1, 0))  # Move right (towards +x)
            elif dy_pos == min(dx_pos, dx_neg, dy_pos, dy_neg):
                moves.append((0, -1)) # Move down (towards -y)
            else:
                moves.append((0, 1))  # Move up (towards +y)
            
            # Add random noise
            noise = np.random.normal(0, temp, size=2)
            # Apply move
            for dx, dy in moves:
                p[idx, 0] += dx * move_magnitude + noise[0]
                p[idx, 1] += dy * move_magnitude + noise[1]
                
        else:
            # Pair is bottleneck
            # Move i and j apart
            vec = p[j] - p[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = np.array([1.0, 0.0])
            
            # Move i in -vec direction, j in +vec direction
            noise_i = np.random.normal(0, temp, size=2)
            noise_j = np.random.normal(0, temp, size=2)
            
            p[i] -= vec * move_magnitude + noise_i
            p[j] += vec * move_magnitude + noise_j
            
            # Also check if moving them hits walls, if so, adjust
            # Simple projection: if out of bounds, pull back
            p[i] = np.clip(p[i], -0.4999, 0.4999)
            p[j] = np.clip(p[j], -0.4999, 0.4999)
            
        # Clip all points to stay inside (safety)
        p = np.clip(p, -0.4999, 0.4999)
        
        return p

    # Initialization: Random points inside the square
    # Better initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    # Add small random jitter to break symmetry
    points += np.random.normal(0, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_score = evaluate(best_points)
    
    # Simulated Annealing parameters
    initial_temp = 0.05
    final_temp = 0.001
    iterations = 5000 # Adjust based on time
    cooling_rate = (final_temp / initial_temp) ** (1.0 / iterations)
    
    temp = initial_temp
    
    # Time limit check
    import time
    start_time = time.time()
    time_limit = 100 # Leave 20s buffer
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        new_points = improve(points, temp)
        new_score = evaluate(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            
        # Acceptance criterion
        # If new_score >= current_score (implicit in best tracking), accept.
        # For SA, we might accept worse moves to escape local minima.
        # However, since we are tracking global best, we can just update current points
        # if the move is "good enough" or with probability exp((new-old)/T).
        
        # Let's use a simpler Hill Climbing with Random Restarts for robustness
        # Actually, pure Hill Climbing on min-dist is prone to local minima.
        # Let's stick to SA logic:
        # Calculate delta
        # We need to track current score to compare with new score for acceptance
        # But `points` is the current state.
        
        # Let's refactor the loop to track current state properly.
        pass
        
    # The previous loop structure was slightly flawed for standard SA because it didn't update `points` based on acceptance.
    # Let's rewrite the optimization loop properly.
    
    points = best_points.copy()
    current_score = best_score
    
    temp = initial_temp
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        new_points = improve(points, temp)
        new_score = evaluate(new_points)
        
        # Accept if better, or with probability exp((new - current) / temp)
        if new_score >= current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            prob = np.exp((new_score - current_score) / temp)
            if np.random.rand() < prob:
                points = new_points
                current_score = new_score
                
        temp *= cooling_rate
        
    return best_points

# EVOLVE-BLOCK-END
