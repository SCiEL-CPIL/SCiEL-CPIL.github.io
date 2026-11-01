import numpy as np
import time

N = 21

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 110.0  # Safety margin for 120s limit
    
    # We pack in a 1x1 square (perimeter 4)
    w, h = 1.0, 1.0
    
    # Initial positions: Grid layout
    # 21 circles. Let's try a 3x7 grid or 5x5 (with 4 missing) or just random.
    # Grid 3x7: spacing 1/3 x 1/7? No, 1x1 box.
    # Let's use a hexagonal-like initial distribution or just random.
    # Random is better to avoid symmetry traps.
    
    np.random.seed(42)
    # Positions
    xs = np.random.uniform(0.2, 0.8, N)
    ys = np.random.uniform(0.2, 0.8, N)
    radii = np.full(N, 0.05)
    
    best_sum_r = np.sum(radii)
    best_xs, best_ys, best_r = xs.copy(), ys.copy(), radii.copy()
    
    # Parameters
    dt = 0.05  # Time step for movement
    dr_step = 0.001 # Radius growth step
    cooling = 1.0
    min_cooling = 0.01
    
    # Main loop
    while time.time() - start_time < max_time:
        
        # 1. Calculate forces (Repulsion from overlaps and walls)
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-Circle Repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Overlap: push apart
                    # Force magnitude proportional to overlap
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += (dx / dist) * force
                    fy[i] += (dy / dist) * force
                    fx[j] -= (dx / dist) * force
                    fy[j] -= (dy / dist) * force
                elif dist <= 1e-6:
                    # Coincident: random push
                    fx[i] += np.random.uniform(-1, 1)
                    fy[i] += np.random.uniform(-1, 1)
                    fx[j] -= np.random.uniform(-1, 1)
                    fy[j] -= np.random.uniform(-1, 1)

        # Wall Repulsion
        for i in range(N):
            # Left wall
            if xs[i] < radii[i]:
                fx[i] += (radii[i] - xs[i]) * 10.0
            # Right wall
            if xs[i] > w - radii[i]:
                fx[i] -= (xs[i] - (w - radii[i])) * 10.0
            # Bottom wall
            if ys[i] < radii[i]:
                fy[i] += (radii[i] - ys[i]) * 10.0
            # Top wall
            if ys[i] > h - radii[i]:
                fy[i] -= (ys[i] - (h - radii[i])) * 10.0

        # 2. Update Positions
        xs += fx * dt * cooling
        ys += fy * dt * cooling
        
        # 3. Try to increase radii
        # Check if system is stable (low overlap)
        max_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    max_overlap = max(max_overlap, min_dist - dist)
        
        # Also check walls
        for i in range(N):
            if xs[i] < radii[i]: max_overlap = max(max_overlap, radii[i] - xs[i])
            if xs[i] > w - radii[i]: max_overlap = max(max_overlap, xs[i] - (w - radii[i]))
            if ys[i] < radii[i]: max_overlap = max(max_overlap, radii[i] - ys[i])
            if ys[i] > h - radii[i]: max_overlap = max(max_overlap, ys[i] - (h - radii[i]))
        
        if max_overlap < 1e-4:
            # Stable, grow radii
            radii += dr_step * cooling
        else:
            # Unstable, shrink radii slightly or just wait
            radii -= dr_step * 0.5 * cooling
            
        # 4. Cooling
        cooling *= 0.999
        if cooling < min_cooling:
            cooling = min_cooling
            # Random restart/jitter if stuck?
            # For now, just keep going with small jitter
            xs += np.random.normal(0, 0.01, N)
            ys += np.random.normal(0, 0.01, N)
            
        # 5. Track Best
        current_sum = np.sum(radii)
        if current_sum > best_sum_r:
            best_sum_r = current_sum
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_r = radii.copy()

    # Construct final array
    # Ensure final result is valid (clamp positions, ensure no overlap? 
    # The evaluator checks validity. We should return the best valid state.
    # To be safe, we can do a final cleanup pass to strictly enforce constraints 
    # on the best solution found.
    
    xs = best_xs
    ys = best_ys
    radii = best_r
    
    # Final cleanup: strictly enforce bounds and resolve any tiny overlaps 
    # by shrinking radii if necessary.
    for _ in range(100):
        valid = True
        # Check walls
        for i in range(N):
            if xs[i] < radii[i]:
                xs[i] = radii[i]
                valid = False
            if xs[i] > w - radii[i]:
                xs[i] = w - radii[i]
                valid = False
            if ys[i] < radii[i]:
                ys[i] = radii[i]
                valid = False
            if ys[i] > h - radii[i]:
                ys[i] = h - radii[i]
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both radii to fit
                    # This is a simple heuristic. Better to move them apart.
                    # But we are at the end. Shrinking guarantees validity.
                    # However, we want to maximize sum. 
                    # The optimization loop should have handled this.
                    # If we are here, it means the "best" had tiny overlaps.
                    # Let's just shrink the smaller one or both slightly.
                    overlap = min_dist - dist
                    # Reduce radii proportionally
                    radii[i] -= overlap / 2.0
                    radii[j] -= overlap / 2.0
                    valid = False
                    # Prevent negative radii
                    if radii[i] < 0: radii[i] = 0
                    if radii[j] < 0: radii[j] = 0
        
        if valid:
            break
            
    return np.column_stack((xs, ys, radii))
