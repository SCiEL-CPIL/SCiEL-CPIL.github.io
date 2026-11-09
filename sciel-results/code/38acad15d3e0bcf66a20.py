"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _simulate(circles, aspect_ratio, t_start, t_end, steps, dt, k_rep, k_wall, k_radius, rng):
    """
    Simulate circle packing using force-directed method.
    circles: (N, 3) array [x, y, r]
    aspect_ratio: w/h
    """
    w = 2.0 * aspect_ratio
    h = 2.0 / aspect_ratio
    
    # Normalize initial positions to fit within [0, w] x [0, h]
    # Assuming input is in [0,1]x[0,1] or similar, but we will re-init mostly.
    # However, if we pass in a grid, we should map it.
    # For this function, we assume we start with a configuration already scaled or we scale here.
    # Let's assume input is in [0,1]x[0,1] and scale to [0,w]x[0,h]
    scaled_circles = np.copy(circles)
    scaled_circles[:, 0] *= w
    scaled_circles[:, 1] *= h
    
    # Initial radii: small
    # We want to maximize sum of radii.
    # Start with small radii and grow them.
    # Or start with max possible non-overlapping radii?
    # Force-directed usually works better if we start small and grow, or start large and relax.
    # Let's start with a radius that fits in a grid, then grow.
    # Grid radius for N=21 in w x h.
    # Simpler: start with r=0.1 and grow.
    
    curr_r = 0.1
    scaled_circles[:, 2] = curr_r
    
    best_sum = 0
    best_state = None
    
    # Precompute indices for pairs
    n = len(scaled_circles)
    indices = np.arange(n)
    
    # Simulation loop
    # We use a simple Euler integration with damping
    velocity = np.zeros((n, 2))
    
    # Time steps
    # We want to run for a fixed number of steps or time
    # "Temperature" controls the strength of forces and random noise
    
    for step in range(steps):
        # Progress 0 to 1
        p = step / steps
        # Cooling schedule: start high (aggressive growth/repulsion), end low (settle)
        temp = t_start * (t_end / t_start) ** p
        
        # Calculate forces
        forces = np.zeros((n, 2))
        
        # 1. Repulsion between circles
        # F = k_rep * (r_i + r_j - dist) / dist * (pos_i - pos_j) if dist < r_i + r_j
        # Or Coulomb: F = k_rep * (r_i + r_j)^2 / dist^2 ?
        # Overlap resolution is better for packing.
        # Let's use a strong repulsive force when overlapping, weak otherwise.
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = scaled_circles[i, 0] - scaled_circles[j, 0]
                dy = scaled_circles[i, 1] - scaled_circles[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                min_dist = scaled_circles[i, 2] + scaled_circles[j, 2]
                
                if dist < 1e-6:
                    dx = 0.01
                    dy = 0.01
                    dist = np.sqrt(0.0002)
                
                if dist < min_dist:
                    # Overlapping
                    # Force proportional to overlap depth
                    overlap = min_dist - dist
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    # Force magnitude
                    f = k_rep * overlap * temp
                    forces[i, 0] += f * nx
                    forces[i, 1] += f * ny
                    forces[j, 0] -= f * nx
                    forces[j, 1] -= f * ny
                else:
                    # Weak repulsion to keep them separated (prevents clustering too tight)
                    # Optional, might slow down. Let's skip for speed unless necessary.
                    pass
        
        # 2. Wall attraction / containment
        # If circle is too close to wall, push it in.
        # Also, we want to push circles to the walls to maximize packing density?
        # Actually, for maximizing radius sum, we want them to fill the space.
        # So, we should push them OUT of the walls (containment) and allow them to grow.
        # If we just contain them, they will settle in the center.
        # To maximize radius, we need to expand the "effective" container or grow radii until constrained.
        
        # Strategy: 
        # 1. Grow radii slightly each step.
        # 2. Apply containment forces (push back if outside).
        # 3. Apply repulsion forces (push apart if overlapping).
        # 4. Integrate.
        
        # Grow radii
        # Growth rate depends on temperature?
        # Let's grow linearly or exponentially towards a max?
        # Max possible radius is bounded by area.
        # Area of rect = 4. Area of circles = N * pi * r^2.
        # Max r ~ sqrt(4 / (N * pi)) ~ sqrt(4/66) ~ 0.24.
        # Let's grow towards 0.25.
        
        # Update radii
        # We can try to increase radius if no overlaps?
        # Simpler: increase radius by a small amount, let forces resolve.
        # If they overlap, forces will push them apart, but if they can't (wall), radius growth is blocked?
        # Actually, if we grow radius, they overlap, force pushes them apart.
        # If they hit wall, they stay at wall.
        # This effectively finds the max radius for the current positions.
        
        # Let's increase radius by a small step
        dr = k_radius * temp
        new_r = curr_r + dr
        # Cap radius
        max_r = 0.5
        if new_r > max_r:
            new_r = max_r
        scaled_circles[:, 2] = new_r
        curr_r = new_r
        
        # Wall Forces
        for i in range(n):
            r = scaled_circles[i, 2]
            # Left
            if scaled_circles[i, 0] < r:
                forces[i, 0] += k_wall * (r - scaled_circles[i, 0]) * temp
            # Right
            if scaled_circles[i, 0] > w - r:
                forces[i, 0] -= k_wall * (scaled_circles[i, 0] - (w - r)) * temp
            # Bottom
            if scaled_circles[i, 1] < r:
                forces[i, 1] += k_wall * (r - scaled_circles[i, 1]) * temp
            # Top
            if scaled_circles[i, 1] > h - r:
                forces[i, 1] -= k_wall * (scaled_circles[i, 1] - (h - r)) * temp
            
            # Random jitter (Annealing)
            forces[i, 0] += rng.uniform(-1, 1) * temp * 0.1
            forces[i, 1] += rng.uniform(-1, 1) * temp * 0.1
        
        # Integrate
        # velocity = velocity * damping + forces * dt
        # damping = 0.9
        velocity = velocity * 0.9 + forces * dt
        scaled_circles[:, 0] += velocity[:, 0]
        scaled_circles[:, 1] += velocity[:, 1]
        
        # Hard constraint: clamp positions to stay within bounds (with margin r)
        # This prevents exploding out
        for i in range(n):
            r = scaled_circles[i, 2]
            if scaled_circles[i, 0] < r: scaled_circles[i, 0] = r
            if scaled_circles[i, 0] > w - r: scaled_circles[i, 0] = w - r
            if scaled_circles[i, 1] < r: scaled_circles[i, 1] = r
            if scaled_circles[i, 1] > h - r: scaled_circles[i, 1] = h - r
        
        # Check validity and score
        # Validity: no overlaps
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = scaled_circles[i, 0] - scaled_circles[j, 0]
                dy = scaled_circles[i, 1] - scaled_circles[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = scaled_circles[i, 2] + scaled_circles[j, 2]
                if dist_sq < min_dist*min_dist:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            sum_r = np.sum(scaled_circles[:, 2])
            if sum_r > best_sum:
                best_sum = sum_r
                best_state = np.copy(scaled_circles)
        
        # Time check
        if time.time() > t_end:
            break
            
    return best_state, best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_result = None
    best_score = 0.0
    
    # Search over aspect ratios
    # Perimeter 4 => 2(w+h)=4 => w+h=2.
    # w in [1, 1.5] is a good range (square to 1.5:1)
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    # Also some random ones
    rng = np.random.default_rng(42)
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = []
    for i in range(side):
        for j in range(side):
            if len(grid_cells) < N:
                grid_cells.append([ (i + 0.5) / side, (j + 0.5) / side, 0.1 ])
    initial_grid = np.array(grid_cells)
    
    # Parameters
    steps = 2000
    dt = 0.01
    k_rep = 10.0
    k_wall = 10.0
    k_radius = 0.001 # Growth rate per step
    t_start = 1.0
    t_end = 0.1
    
    # Try different aspect ratios
    for ar in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
            
        # Run simulation
        # We might want to restart a few times for each aspect ratio
        restarts = 3
        for r in range(restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb initial grid
            perturbed = np.copy(initial_grid)
            if r > 0:
                # Add noise
                noise = rng.uniform(-0.1, 0.1, (N, 2))
                perturbed[:, 0] += noise[:, 0]
                perturbed[:, 1] += noise[:, 1]
                # Clamp to [0,1]
                perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
                perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
            
            # Run simulation
            # We need to pass a fresh rng or use global?
            # Using local rng for reproducibility per restart
            local_rng = np.random.default_rng(100 + r)
            
            result, score = _simulate(
                perturbed, ar, 
                time.time(), deadline, 
                steps, dt, k_rep, k_wall, k_radius, local_rng
            )
            
            if result is not None and score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Rescale best_result to fit perimeter 4?
    # The simulation was done in a box of size w x h where 2(w+h)=4.
    # So the coordinates are already in the correct scale.
    # x in [0, w], y in [0, h].
    # The evaluator expects a bounding rectangle with perimeter <= 4.
    # Our box has perimeter 4.
    # We just need to return the circles.
    
    return best_result
# EVOLVE-BLOCK-END
