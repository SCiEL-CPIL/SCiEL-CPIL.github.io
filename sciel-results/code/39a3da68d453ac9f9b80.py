"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin for 120s limit
    
    # Initial grid setup for 26 circles
    # A 5x5 grid gives 25 positions. We need 26.
    # Let's use a 5x5 grid and place the 26th circle in the center of one of the cells, 
    # or better, just use a grid of side ceil(sqrt(26)) = 6? No, 6x6=36 is too sparse.
    # 5x5 grid: 25 spots. 
    # Let's try a 5x5 grid with slightly smaller circles to fit, and then optimize.
    # Or better: initialize with 26 circles in a grid-like pattern.
    # 26 is close to 25. Let's use a 5x5 grid for the first 25, and place the 26th 
    # in the middle, slightly overlapping initially, but we will fix overlaps in SA.
    # Actually, it's safer to start with a valid configuration.
    # Let's use a 5x5 grid of 25 circles, and add one more.
    # To ensure validity at start, we can make them very small.
    # But SA works best starting from a reasonable size.
    # Let's initialize with a 5x5 grid of radius 0.19 (diameter 0.38, fits in 0.5 cell).
    # And the 26th circle also small.
    
    # Better initialization: 
    # Use a grid of 5x5. 
    # Positions: (i+0.5)/5, (j+0.5)/5 for i,j in 0..4.
    # Radius: 0.5/5 - epsilon = 0.1 - 0.01 = 0.09.
    # This is a very small radius. Sum = 25 * 0.09 = 2.25.
    # We can do much better.
    
    # Let's try a different initialization:
    # Place 26 circles in a random valid configuration? Hard.
    # Let's stick to the grid but use a denser packing heuristic for init.
    # 26 circles. 
    # 5 columns, 5 rows = 25. One extra.
    # Let's just use a 6x6 grid and pick 26 spots? 
    # 6x6 grid: cell size 1/6. Radius ~ 1/12 = 0.083.
    # Sum = 26 * 0.083 = 2.16.
    
    # Actually, let's just start with 26 circles in a grid pattern that is valid.
    # We can use a 5x5 grid and put the 26th in the center, but shrink all to fit.
    # Or simpler: Start with 26 circles, all radius 0.05, placed randomly in [0.05, 0.95]^2.
    # Then use SA to grow them.
    
    # Random initialization is often fine for SA if the temperature is high enough.
    # But to be safe, let's start with a grid.
    # 26 circles. 
    # Let's use a 5x5 grid for 25 circles, and place the 26th at (0.5, 0.5) with small radius.
    # Then shrink all radii to ensure no overlap.
    
    side = 5
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([(i + 0.5) / side, (j + 0.5) / side])
    # Add 26th circle
    coords.append([0.5, 0.5])
    
    # Initial radius: very small to ensure validity
    r_init = 0.01
    circles = np.array([(c[0], c[1], r_init) for c in coords])
    
    # We will increase radii during optimization.
    
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    max_steps = 500000
    step = 0
    
    while step < max_steps:
        if time.time() > deadline:
            break
            
        if step % 1000 == 0:
            T *= cooling
        
        # Pick a random circle
        idx = np.random.randint(N)
        
        # Decide move type: position or radius
        if np.random.rand() < 0.8:
            # Move position
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            new_x = circles[idx, 0] + dx
            new_y = circles[idx, 1] + dy
            new_r = circles[idx, 2]
        else:
            # Change radius
            new_x = circles[idx, 0]
            new_y = circles[idx, 1]
            # Try to increase radius mostly
            if np.random.rand() < 0.7:
                new_r = circles[idx, 2] + np.random.uniform(0, 0.02)
            else:
                new_r = circles[idx, 2] - np.random.uniform(0, 0.02)
                if new_r < 0.01: new_r = 0.01
        
        # Create new candidate
        new_circles = circles.copy()
        new_circles[idx, 0] = new_x
        new_circles[idx, 1] = new_y
        new_circles[idx, 2] = new_r
        
        # Check validity
        valid = True
        # Check bounds
        if new_x < new_r or new_x > 1 - new_r or new_y < new_r or new_y > 1 - new_r:
            valid = False
        
        if valid:
            # Check overlaps with other circles
            for j in range(N):
                if j == idx:
                    continue
                dx2 = new_circles[idx, 0] - new_circles[j, 0]
                dy2 = new_circles[idx, 1] - new_circles[j, 1]
                dist2 = dx2*dx2 + dy2*dy2
                min_dist = new_circles[idx, 2] + new_circles[j, 2]
                if dist2 < min_dist*min_dist:
                    valid = False
                    break
        
        new_score = np.sum(new_circles[:, 2])
        
        if valid:
            if new_score > best_score:
                best_score = new_score
                best_circles = new_circles.copy()
            
            delta = new_score - np.sum(circles[:, 2])
            if delta > 0:
                circles = new_circles
            else:
                # Accept with probability exp(delta / T)
                if np.random.rand() < np.exp(delta / T):
                    circles = new_circles
        else:
            # If invalid, reject (or penalize heavily, but simple rejection is safer for stability)
            # We can also try to repair, but rejection is standard for simple SA
            pass
            
        step += 1
        
    return best_circles
# EVOLVE-BLOCK-END
