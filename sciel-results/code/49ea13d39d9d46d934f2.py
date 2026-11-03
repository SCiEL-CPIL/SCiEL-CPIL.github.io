import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_score = 0.0
    best_positions = None
    
    # Helper to calculate score (minimum distance scaled)
    def calc_score(pos):
        # pos is (N, 2)
        # Calculate pairwise distances
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist = np.sqrt(np.sum(diff**2, axis=-1))
        # This creates an N x N matrix.
        # We only need upper triangle for i<j
        # But vectorized full matrix is faster in NumPy usually.
        
        # Boundary distance
        dist_center = np.sqrt(np.sum(pos**2, axis=1))
        dist_wall = 1.0 - dist_center
        
        # Pairwise distances
        # Using broadcasting
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist_sq = np.min(dist_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # The radius is min(min_pair_dist/2, min_dist_wall)
        # Score is proportional to this radius.
        # Let's maximize min(min_pair_dist, 2*min_dist_wall)
        score = min(min_pair_dist, 2.0 * np.min(dist_wall))
        return score

    # Initialization: Hexagonal Lattice
    def init_hex():
        # Create a hexagonal lattice
        # Spacing s. We want to fit ~50 circles.
        # Area of unit circle ~ 3.14. Area of 50 circles of radius r ~ 50 * pi * r^2.
        # 50 * r^2 ~ 3.14 / pi ~ 1 => r ~ 0.14.
        # Hex spacing s = 2r ~ 0.28.
        # Let's try a range of spacings or just a dense lattice and pick points.
        
        # Generate a grid of hex points
        s = 0.25 # Approximate spacing
        # Generate points in a square area covering the unit circle
        # Range -1.5 to 1.5
        xs = np.arange(-1.5, 1.5, s/2) # Half step for hex offset
        # Actually, standard hex generation:
        # Row j, Col i
        # x = (i + 0.5*(j%2)) * s
        # y = j * s * sqrt(3)/2
        
        points = []
        max_k = int(1.5 / (s * np.sqrt(3)/2)) + 2
        for j in range(-max_k, max_k+1):
            for i in range(-max_k, max_k+1):
                x = (i + 0.5*(j%2)) * s
                y = j * s * np.sqrt(3)/2
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback to random if lattice is too sparse (unlikely)
            return np.random.uniform(-0.8, 0.8, (N, 2))
        
        # Select N points from the lattice
        # Strategy: Greedy farthest point sampling from the lattice points
        # Start with a point close to center, then pick the lattice point farthest from existing set
        # This ensures good spread.
        
        # Find point closest to center
        d_centers = np.sum(points**2, axis=1)
        start_idx = np.argmin(d_centers)
        selected = [points[start_idx]]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)
        
        for _ in range(N-1):
            # Calculate min distance from each remaining point to selected set
            # selected is list of arrays
            # Vectorized:
            sel_arr = np.array(selected) # (k, 2)
            rem_arr = points[remaining] # (m, 2)
            
            # Distances between rem and sel
            # (m, 1, 2) - (1, k, 2) -> (m, k, 2)
            diff = rem_arr[:, np.newaxis, :] - sel_arr[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1) # (m, k)
            min_dists_sq = np.min(dists_sq, axis=1) # (m,)
            
            # Pick the one with max min_dist
            best_idx_in_rem = np.argmax(min_dists_sq)
            best_idx_in_full = remaining[best_idx_in_rem]
            selected.append(points[best_idx_in_full])
            remaining.remove(best_idx_in_full)
            
            if not remaining:
                break
                
        return np.array(selected)

    def run_sa(pos, time_limit):
        pos = pos.copy()
        current_score = calc_score(pos)
        best_score_local = current_score
        best_pos_local = pos.copy()
        
        T = 0.05 # Initial temperature
        T_min = 0.001
        cooling = 0.9999 # Slow cooling
        
        # Move magnitude
        step_size = 0.05
        
        while time.time() < time_limit:
            # Pick a random circle to move
            idx = np.random.randint(N)
            
            # Generate a random move
            # Gaussian noise
            move = np.random.normal(0, step_size, 2)
            new_pos = pos.copy()
            new_pos[idx] += move
            
            # Check if inside boundary (hard constraint? No, SA can handle soft)
            # But if it goes way outside, score drops significantly.
            
            new_score = calc_score(new_pos)
            
            if new_score > current_score:
                pos = new_pos
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_pos_local = pos.copy()
            else:
                # Accept with probability exp((new - old) / T)
                # Since we maximize, delta = new - old is negative.
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        pos = new_pos
                        current_score = new_score
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Occasionally reheat or reset step size?
            # Keep it simple.
            
        return best_pos_local, best_score_local

    # Run multiple restarts
    # First run with hex init
    pos_init = init_hex()
    # Ensure init is valid (inside circle)
    # If any point is outside, project it? Or just let SA fix it.
    # The score function penalizes being outside (dist_wall becomes negative, 2*dist_wall negative, min becomes negative).
    # So SA will push them in.
    
    # Give first run a bit more time? Or equal split?
    # Let's do 3 restarts.
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            pos_current = pos_init
        else:
            # Perturb the best found so far for diversity
            if best_positions is not None:
                pos_current = best_positions + np.random.normal(0, 0.1, (N, 2))
            else:
                pos_current = init_hex()
        
        # Run SA for this restart
        t_end = min(deadline, time.time() + time_per_restart)
        # Ensure at least 1 second per run
        if t_end - time.time() < 1.0:
             t_end = time.time() + 1.0
             
        pos_res, score_res = run_sa(pos_current, t_end)
        
        if score_res > best_score:
            best_score = score_res
            best_positions = pos_res

    # If no valid position found (shouldn't happen), return a simple valid config
    if best_positions is None:
        # Fallback: Hex lattice cut
        s = 0.25
        points = []
        max_k = 10
        for j in range(-max_k, max_k+1):
            for i in range(-max_k, max_k+1):
                x = (i + 0.5*(j%2)) * s
                y = j * s * np.sqrt(3)/2
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Sort by distance to center, take first N
            d_centers = np.sum(points**2, axis=1)
            idx = np.argsort(d_centers)[:N]
            best_positions = points[idx]
        else:
            # Random
            best_positions = np.random.uniform(-0.5, 0.5, (N, 2))

    return best_positions
