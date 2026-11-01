import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # Start with a grid-like initial guess
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small jitter to break symmetry
    centers += np.random.normal(0, 0.01, (N, 3))
    centers = np.clip(centers, -0.49, 0.49)
    
    def calc_radius(centers):
        wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
        r_wall = np.min(wall_dist)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_wall, r_pair)

    current_r = calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # MC Parameters
    T = 0.02
    T_min = 1e-6
    cooling_rate = 0.9999
    step_size = 0.03
    
    # Phase 1: Simulated Annealing
    while time.time() - start_time < timeout - 10.0:
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 3)
        
        new_center = centers[idx] + delta
        
        if np.any(new_center < -0.5) or np.any(new_center > 0.5):
            continue
            
        centers[idx] = new_center
        new_r = calc_radius(centers)
        
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -T * 10:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_r = new_r
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = centers.copy()
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat if stuck? Simple reheat
            if np.random.random() < 0.1:
                T = 0.01
                step_size = 0.03

    # Phase 2: Local Polish
    polish_start = time.time()
    polish_time_limit = timeout - (polish_start - start_time) - 2.0
    if polish_time_limit > 1.0:
        current_best = best_centers.copy()
        current_best_r = best_r
        step = 0.02
        
        while time.time() - polish_start < polish_time_limit:
            improved = False
            for i in range(N):
                orig = current_best[i].copy()
                # Try several small moves
                for _ in range(20):
                    delta = np.random.normal(0, step, 3)
                    new_c = current_best[i] + delta
                    if np.any(new_c < -0.5) or np.any(new_c > 0.5):
                        continue
                    current_best[i] = new_c
                    r = calc_radius(current_best)
                    if r > current_best_r:
                        current_best_r = r
                        improved = True
                    else:
                        current_best[i] = orig
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
            # Update global best
            if current_best_r > best_r:
                best_r = current_best_r
                best_centers = current_best.copy()
        
        best_centers = current_best
        best_r = current_best_r

    # Final validation
    if np.any(best_centers < -0.5) or np.any(best_centers > 0.5):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
