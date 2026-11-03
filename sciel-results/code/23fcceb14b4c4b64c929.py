import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_r = 0.0
    best_centers = None
    
    # ... helper functions ...
    
    def run_optimization(centers, steps, time_limit):
        current_r = calc_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        for step in range(steps):
            if time.time() > time_limit:
                break
                
            current_r = calc_radius(centers)
            if current_r > best_local_r:
                best_local_r = current_r
                best_local_centers = centers.copy()
                
            progress = step / steps
            step_size = 0.05 * (1.0 - progress) + 0.001
            noise_scale = 0.05 * (1.0 - progress) + 0.001
            
            new_centers = gradient_step(centers, current_r, step_size, noise_scale)
            
            new_r = calc_radius(new_centers)
            
            # Accept if better, or with some probability if worse (SA)
            if new_r > current_r:
                centers = new_centers
                current_r = new_r
            else:
                # Simple SA acceptance
                # T decreases
                T = 0.01 * (1.0 - progress) + 0.0001
                if np.random.rand() < np.exp(-(current_r - new_r) / T):
                    centers = new_centers
                    current_r = new_r
        return best_local_r, best_local_centers

    # Initial guess
    # Try a few different initializations
    inits = []
    
    # 1. Random
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
    
    # 2. Grid + 1
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    # Add 50th point in a gap
    # Center of a cell, e.g., between (0,0) and (1/7, 0) -> (1/14, 0)
    # But (0,0) is in grid.
    # Let's just add a random point for now, optimization will fix it
    grid_50 = np.vstack([grid_49, [0.1, 0.1]])
    inits.append(grid_50)
    
    # 3. Hexagonal-ish
    # Just random for now, grid is covered by init 2
    
    # Run first optimization to gauge speed
    t_start = time.time()
    r0, c0 = run_optimization(inits[0], 500, deadline)
    t_elapsed = time.time() - t_start
    
    best_r = r0
    best_centers = c0
    
    # Estimate time per step
    # 500 steps took t_elapsed
    # Time per step approx
    t_per_step = t_elapsed / 500.0
    
    # How many more steps can we do?
    time_remaining = deadline - time.time()
    steps_per_restart = int(2000) # Target steps per restart
    
    # If we have time for more restarts
    num_restarts_possible = int(time_remaining / (steps_per_restart * t_per_step))
    
    # Cap restarts
    max_restarts = 10
    num_restarts = min(num_restarts_possible, max_restarts)
    
    # If we can't even do 1 more, just polish the current one
    if num_restarts == 0:
         # Just do a final polish on best_centers
         # ...
         pass
    else:
        # Run more restarts
        # We already did 1 (init 0).
        # We have inits[1] left.
        # We can also generate new random inits.
        
        for i in range(1, num_restarts + 1):
            if time.time() > deadline:
                break
                
            if i < len(inits):
                init_c = inits[i]
            else:
                init_c = np.random.uniform(-0.4, 0.4, size=(N, 2))
                
            # Run optimization
            # Use remaining time / remaining restarts
            time_slice = (deadline - time.time()) / (num_restarts - i + 1)
            steps = int(time_slice / t_per_step)
            if steps < 100:
                steps = 100
                
            r, c = run_optimization(init_c, steps, time.time() + time_slice)
            
            if r > best_r:
                best_r = r
                best_centers = c

    # Final Polish
    if best_centers is not None and time.time() < deadline - 2.0:
        centers = best_centers
        current_r = calc_radius(centers)
        # Small steps, low noise
        for _ in range(500):
            if time.time() > deadline - 1.0:
                break
            current_r = calc_radius(centers)
            step_size = 0.002
            noise_scale = 0.0005
            new_centers = gradient_step(centers, current_r, step_size, noise_scale)
            new_r = calc_radius(new_centers)
            if new_r > current_r:
                centers = new_centers
                current_r = new_r
                best_r = current_r
                best_centers = centers.copy()

    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.4, 0.4, size=(N, 2))

    return best_centers
