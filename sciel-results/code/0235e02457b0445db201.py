"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START

def _simulate_packing(n_circles, time_budget):
    """
    Simulate a granular assembly of circles in a unit square.
    Circles start in a dense hexagonal lattice with overlap.
    They expand and push each other apart until equilibrium.
    """
    start_time = time.time()
    
    # 1. Initial Setup: Hexagonal Lattice
    # Arrange circles in a hexagonal close-packed structure.
    # This is a dense starting point.
    cols = int(np.ceil(np.sqrt(n_circles * np.sqrt(3) / 2)))
    rows = int(np.ceil(n_circles / cols))
    
    # Calculate spacing to ensure overlap initially (dense start)
    # Hexagonal spacing: dx = r, dy = r * sqrt(3)
    # We want them to overlap, so we start with a small radius.
    r0 = 0.1 
    dx = r0 * 0.9  # Slight overlap in x
    dy = r0 * np.sqrt(3) * 0.9 # Slight overlap in y
    
    # Generate positions
    positions = []
    radii = np.full(n_circles, r0)
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n_circles:
                break
            x = 0.5 + col * dx + (row % 2) * (dx / 2.0)
            y = 0.5 + row * dy
            positions.append([x, y])
            count += 1
    
    # If we didn't get enough, fill remaining randomly or adjust grid
    # But for N=32, a 6x6 grid (36 spots) is enough.
    positions = np.array(positions[:n_circles])
    
    # Center the configuration in the square
    # Find bounding box
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale and shift to center
    # We want to keep the aspect ratio but center it.
    # Actually, let's just center the mean position at (0.5, 0.5)
    mean_pos = positions.mean(axis=0)
    positions -= mean_pos
    positions += 0.5
    
    velocities = np.zeros_like(positions)
    
    # Simulation Parameters
    dt = 0.005
    max_steps = 20000
    damping = 0.95  # Velocity damping
    k_overlap = 50.0 # Stiffness for overlap penalty
    k_wall = 50.0    # Stiffness for wall penalty
    expansion_rate = 0.0001 # How fast radii grow
    
    best_score = -1.0
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget - 5.0:
            break
            
        forces = np.zeros_like(positions)
        
        # 1. Expansion Force (Internal Pressure)
        # This simulates the "pressure" trying to expand the assembly.
        # We model this as a force pushing centers away from the global center of mass?
        # Or simply increasing radii?
        # The prompt says "expand their radii under a constant internal pressure".
        # In a rigid body sim, if radii are fixed, we can't expand them.
        # So we treat 'radius' as a variable that grows, and the force is the interaction.
        # However, standard circle packing optimization often fixes radii and moves centers, 
        # OR fixes centers and grows radii.
        # The "granular assembly" metaphor usually implies fixed radii and moving centers to pack.
        # BUT the goal is to MAXIMIZE sum of radii.
        # Strategy: 
        # 1. Start with small radii.
        # 2. Grow radii slowly.
        # 3. Use repulsive forces to maintain non-overlap.
        # 4. The "pressure" is the growth of radii itself.
        
        # Update Radii
        radii += expansion_rate
        # Cap radii to prevent explosion if no constraint is hit (shouldn't happen in unit square)
        radii = np.minimum(radii, 0.5)
        
        # 2. Calculate Forces
        # Pairwise Repulsion (Overlap)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    if dist < 1e-6:
                        # Random nudge if exactly overlapping
                        forces[i, 0] += (np.random.rand() - 0.5) * 0.1
                        forces[i, 1] += (np.random.rand() - 0.5) * 0.1
                        forces[j, 0] -= (np.random.rand() - 0.5) * 0.1
                        forces[j, 1] -= (np.random.rand() - 0.5) * 0.1
                    else:
                        overlap = min_dist - dist
                        # Force magnitude proportional to overlap
                        f_mag = k_overlap * overlap
                        # Normal vector
                        nx = dx / dist
                        ny = dy / dist
                        forces[i, 0] += f_mag * nx
                        forces[i, 1] += f_mag * ny
                        forces[j, 0] -= f_mag * nx
                        forces[j, 1] -= f_mag * ny
        
        # Wall Constraints
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            
            # Left Wall
            if x < r:
                overlap = r - x
                forces[i, 0] += k_wall * overlap
            # Right Wall
            if x > 1 - r:
                overlap = x - (1 - r)
                forces[i, 0] -= k_wall * overlap
            # Bottom Wall
            if y < r:
                overlap = r - y
                forces[i, 1] += k_wall * overlap
            # Top Wall
            if y > 1 - r:
                overlap = y - (1 - r)
                forces[i, 1] -= k_wall * overlap
        
        # 3. Update Physics
        # Acceleration = Force / Mass (Mass = 1)
        velocities += forces * dt
        velocities *= damping
        positions += velocities * dt
        
        # 4. Hard Clamp (Safety Net)
        # Ensure circles stay inside square even if integration is unstable
        for i in range(n_circles):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1 - r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1 - r)
            
        # 5. Check Validity and Score
        # A configuration is valid if no overlaps and inside bounds.
        # Due to hard clamping, bounds are always satisfied.
        # We must check overlaps.
        valid = True
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Stop if we've reached a very high score or stagnated
        if best_score > 1.05: # Heuristic: good packing
            # We can try to polish, but let's continue for a bit to see if it grows
            pass
            
    # Final Polish: Local Search on the best valid configuration
    if best_score > 0:
        # Try to push radii up slightly more with a gradient step on the best config
        # This handles the "rigid body" finding the shape, but maybe we can squeeze more.
        pos = best_positions.copy()
        rad = best_radii.copy()
        
        # Simple local gradient ascent on radii?
        # Actually, just return the best found. The simulation usually converges well.
        
        result = np.zeros((n_circles, 3))
        result[:, 0] = pos[:, 0]
        result[:, 1] = pos[:, 1]
        result[:, 2] = rad
        
        return result, best_score
    else:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Run multiple simulations with different random seeds or parameters
    # to escape local minima.
    best_result = None
    best_score = 0.0
    
    np.random.seed(42)
    
    # Try different expansion rates or initial densities
    configs = [
        {'expansion_rate': 0.0001, 'k_overlap': 50.0, 'k_wall': 50.0, 'seed': 1},
        {'expansion_rate': 0.0002, 'k_overlap': 100.0, 'k_wall': 100.0, 'seed': 2},
        {'expansion_rate': 0.00005, 'k_overlap': 20.0, 'k_wall': 20.0, 'seed': 3},
    ]
    
    for cfg in configs:
        if time.time() - start_time > deadline - 10:
            break
            
        np.random.seed(cfg['seed'])
        
        # Modify the simulation to use these params
        # Since _simulate_packing has hardcoded params, we need to pass them or refactor.
        # Let's refactor _simulate_packing to accept params.
        
        res, score = _simulate_packing_with_params(N, time_budget=deadline - (time.time() - start_time) - 5, **cfg)
        
        if res is not None and score > best_score:
            best_score = score
            best_result = res
            
    # Fallback: If simulation failed, use a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result

def _simulate_packing_with_params(n_circles, time_budget, expansion_rate, k_overlap, k_wall, seed):
    """
    Parametric version of the simulation.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # 1. Initial Setup: Hexagonal Lattice
    cols = int(np.ceil(np.sqrt(n_circles * np.sqrt(3) / 2)))
    rows = int(np.ceil(n_circles / cols))
    
    r0 = 0.1 
    dx = r0 * 0.9  
    dy = r0 * np.sqrt(3) * 0.9 
    
    positions = []
    radii = np.full(n_circles, r0)
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n_circles:
                break
            x = 0.5 + col * dx + (row % 2) * (dx / 2.0)
            y = 0.5 + row * dy
            positions.append([x, y])
            count += 1
    
    positions = np.array(positions[:n_circles])
    
    # Add small random jitter to break symmetry
    positions += np.random.uniform(-0.01, 0.01, positions.shape)
    
    # Center
    mean_pos = positions.mean(axis=0)
    positions -= mean_pos
    positions += 0.5
    
    velocities = np.zeros_like(positions)
    
    dt = 0.005
    max_steps = 20000
    damping = 0.95
    
    best_score = -1.0
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget - 2.0:
            break
            
        forces = np.zeros_like(positions)
        
        # 1. Expansion
        radii += expansion_rate
        radii = np.minimum(radii, 0.5)
        
        # 2. Forces
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    if dist < 1e-6:
                        forces[i, 0] += (np.random.rand() - 0.5) * 0.1
                        forces[i, 1] += (np.random.rand() - 0.5) * 0.1
                        forces[j, 0] -= (np.random.rand() - 0.5) * 0.1
                        forces[j, 1] -= (np.random.rand() - 0.5) * 0.1
                    else:
                        overlap = min_dist - dist
                        f_mag = k_overlap * overlap
                        nx = dx / dist
                        ny = dy / dist
                        forces[i, 0] += f_mag * nx
                        forces[i, 1] += f_mag * ny
                        forces[j, 0] -= f_mag * nx
                        forces[j, 1] -= f_mag * ny
        
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            
            if x < r:
                forces[i, 0] += k_wall * (r - x)
            if x > 1 - r:
                forces[i, 0] -= k_wall * (x - (1 - r))
            if y < r:
                forces[i, 1] += k_wall * (r - y)
            if y > 1 - r:
                forces[i, 1] -= k_wall * (y - (1 - r))
        
        # 3. Update
        velocities += forces * dt
        velocities *= damping
        positions += velocities * dt
        
        # 4. Hard Clamp
        for i in range(n_circles):
            r = radii[i]
            positions[i, 0] = np.clip(positions[i, 0], r, 1 - r)
            positions[i, 1] = np.clip(positions[i, 1], r, 1 - r)
            
        # 5. Check
        valid = True
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_radii = radii.copy()
                
        # Early exit if stable and high score
        if best_score > 1.05 and step > 500:
            # Check if radii stopped growing significantly
            # If not growing, we are likely at local max for this config
            if np.all(radii >= best_radii - 1e-6):
                 # Just continue for a bit more to be safe, or break if we've been stuck
                 pass

    if best_score > 0:
        result = np.zeros((n_circles, 3))
        result[:, 0] = best_positions[:, 0]
        result[:, 1] = best_positions[:, 1]
        result[:, 2] = best_radii
        return result, best_score
    else:
        return None, 0.0

# EVOLVE-BLOCK-END
