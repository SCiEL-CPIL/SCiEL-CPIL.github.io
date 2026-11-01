"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    # Safety margin of 10 seconds
    deadline = start_time + 110.0

    # Helper to compute the minimum radius
    def get_radius(p):
        # p: (N, 2) array
        # 1. Pairwise distances
        # Use broadcasting for vectorized distance calculation
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # 2. Boundary distances
        # Distance to closest boundary for each point
        # Square is [-0.5, 0.5]
        bound_dists = np.minimum(0.5 - np.abs(p[:, 0]), 0.5 - np.abs(p[:, 1]))
        min_bound = np.min(bound_dists)
        
        return min(min_pair / 2.0, min_bound)

    # Helper to generate a hexagonal lattice initialization
    def hex_init():
        # Approximate spacing for N points in unit square
        d = np.sqrt(1.0 / (N * 0.866))
        pts = []
        y = -0.5
        row = 0
        while y <= 0.5:
            x_offset = (d / 2) if row % 2 else 0
            x = -0.5 + x_offset
            while x <= 0.5:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
                x += d
            y += d * np.sqrt(3) / 2
            row += 1
            
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback to grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pos = grid[:N].copy()
        else:
            np.random.shuffle(pts)
            pos = pts[:N].copy()
            
        # Ensure strictly inside to avoid initial boundary issues
        pos = np.clip(pos, -0.49, 0.49)
        return pos

    best_overall_r = 0.0
    best_overall_pos = None

    # Multi-restart Simulated Annealing
    # We split the time budget into several restarts to explore different basins
    num_restarts = 4
    time_per_restart = (deadline - start_time) / num_restarts

    for k in range(num_restarts):
        # Initialize position
        if k == 0:
            current_pos = hex_init()
        else:
            # Perturb the best solution found so far
            if best_overall_pos is not None:
                current_pos = best_overall_pos + np.random.normal(0, 0.01, (N, 2))
                current_pos = np.clip(current_pos, -0.49, 0.49)
            else:
                current_pos = hex_init()
        
        current_r = get_radius(current_pos)
        best_local_r = current_r
        best_local_pos = current_pos.copy()
        
        # SA Parameters
        T = 0.01
        T_min = 1e-5
        cooling = 0.999
        step_size = 0.01
        
        stagnant_count = 0
        stagnation_limit = 5000
        
        # End time for this restart
        end_time = start_time + (k + 1) * time_per_restart
        
        while time.time() < end_time:
            if T < T_min:
                break
                
            # Generate a random move
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, 2)
            
            new_pos = current_pos.copy()
            new_pos[i] += delta
            
            new_r = get_radius(new_pos)
            
            # Metropolis acceptance criterion
            if new_r >= current_r or np.random.rand() < np.exp((new_r - current_r) / T):
                current_pos = new_pos
                current_r = new_r
                
                if current_r > best_local_r:
                    best_local_r = current_r
                    best_local_pos = current_pos.copy()
                    stagnant_count = 0
                else:
                    stagnant_count += 1
            else:
                stagnant_count += 1
                
            T *= cooling
            
            # Restart temperature if stuck
            if stagnant_count > stagnation_limit:
                T = 0.005
                stagnant_count = 0
                
        # Update global best
        if best_local_r > best_overall_r:
            best_overall_r = best_local_r
            best_overall_pos = best_local_pos.copy()

    # If no valid solution was found (should not happen), fallback to grid
    if best_overall_pos is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_overall_pos
# EVOLVE-BLOCK-END
