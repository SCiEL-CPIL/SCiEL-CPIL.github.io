"""
Packing 50 circles in a unit square using a Physics-Based Soft-Body Simulation.

Strategy:
Instead of gradient descent on a non-smooth objective function (min distance), 
we simulate the circles as particles with short-range repulsive forces.
1. Initialize with a random or grid configuration.
2. Apply repulsive forces between circles that are closer than a target distance.
3. Apply repulsive forces from the boundaries.
4. Use Velocity Verlet integration with damping.
5. Implement a simple annealing schedule: start with a high "temperature" (random jitter)
   to escape local minima, then cool down to settle into a stable dense packing.
6. The final radius is determined by the minimum distance / 2 and boundary distance.

This method is robust because the force field is smooth (unlike the raw min-distance metric)
and the stochastic nature of the simulation helps explore the configuration space more globally
than deterministic local optimizers.
"""

import numpy as np
import time
import random

N = 50

def _compute_radius(centers):
    """
    Compute the maximum common radius for a given set of centers.
    Returns 0.0 if invalid.
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds (with small tolerance for numerical error)
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    # Clip to valid range for calculation
    centers = np.clip(centers, -0.5, 0.5)
    
    # 1. Boundary distance
    # Distance to each wall
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # 2. Pairwise distance
    # We only need the minimum pairwise distance.
    # For N=50, O(N^2) is 2500 ops, which is very fast.
    min_pairwise = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_pairwise:
                min_pairwise = dist
    
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _simulate(centers_init, seed=42, time_limit=10.0):
    """
    Run a physics simulation to optimize the packing.
    """
    rng = np.random.default_rng(seed)
    
    # State
    pos = centers_init.copy()
    vel = np.zeros_like(pos)
    
    # Parameters
    # We want to push circles apart if they are closer than a certain distance.
    # The optimal radius is ~0.0713. Diameter ~0.1427.
    # Let's set a "soft" repulsion range.
    # If dist < R_repel, force = k * (R_repel - dist)
    # We want the final stable state to have dist >= 2*r.
    # So we can set R_repel slightly larger than the expected diameter to allow "breathing".
    # However, a better approach for packing is to maximize the minimum distance.
    # We can use a force that is zero if dist > d_target, and pushes apart if dist < d_target.
    # But d_target is unknown. 
    # Alternative: Use a generic repulsive potential like Lennard-Jones or simple inverse square.
    # Let's use a simple repulsive force: F = k / d^2 if d < d_cut, else 0.
    # And a boundary force: F = k_b * (0.5 - |x|) if inside, pushing out if too close?
    # Actually, boundary force should push IN if outside, or just constrain.
    # Let's use a potential energy approach:
    # U = sum_{i<j} (1/d_ij)^12  (hard repulsion) + sum_i (1/d_boundary_i)^12
    # This is computationally expensive but N=50 is small.
    # However, gradients of 1/d^12 are very steep.
    
    # Let's stick to a simpler, tunable force model:
    # Repulsion between circles:
    # If d < d_ref: F = k_rep * (d_ref - d) / d  (unit vector direction)
    # We don't know d_ref. 
    # Strategy: Start with a small d_ref and increase it? No, that's not how physics works.
    
    # Better Strategy: 
    # 1. Start with a configuration.
    # 2. Apply forces that push all circles apart as much as possible.
    # 3. The "pressure" of the system will settle at a point where all "gaps" are roughly equal? No, not necessarily.
    
    # Let's use a specific potential:
    # U_pair = (sigma / d)^12
    # U_wall = (sigma_w / d_wall)^12
    # We need to choose sigma.
    # If we fix sigma, the system will settle at a specific density.
    # To maximize radius, we effectively want to find the configuration that minimizes the potential
    # for a given sigma, but we want the sigma that allows the largest min-distance.
    
    # Actually, a simpler approach used in packing algorithms is:
    # 1. Start with random positions.
    # 2. Iteratively:
    #    a. Find the closest pair (i, j) and the closest wall.
    #    b. Move i and j apart, and move the wall-circle pair apart.
    #    This is "Relaxation" or "Jiggling".
    
    # Let's implement a "Jiggling" / "Relaxation" algorithm which is more deterministic and effective for packing.
    # Algorithm:
    # 1. Initialize positions.
    # 2. Loop:
    #    a. Calculate all pairwise distances and boundary distances.
    #    b. Find the minimum distance d_min.
    #    c. If d_min > current_best_r * 2, update best.
    #    d. For the pair (i, j) with min distance, move them apart slightly.
    #    e. For the point k with min boundary distance, move it inward slightly.
    #    f. Add random noise (temperature).
    #    g. Decay temperature.
    
    start_time = time.time()
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Simulation parameters
    dt = 0.01
    damping = 0.9
    # "Temperature" for random kicks
    T = 0.05
    T_decay = 0.999
    T_min = 1e-6
    
    # Force parameters
    # We will use a "soft" repulsion.
    # F = k * (r0 - d) if d < r0 else 0
    # We don't know r0. 
    # Instead, let's use a potential U = 1/d^2.
    # Force F = 2/d^3 * unit_vec.
    # This is always repulsive.
    # The system will expand until constrained by boundaries or other circles.
    # Since the box is fixed, the circles will push against each other and the walls.
    # The equilibrium will be a dense packing.
    
    # To avoid numerical instability with 1/d^3, we clip d.
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2)
                if d < 1e-6:
                    # Random kick to break symmetry
                    forces[i, 0] += rng.normal(0, 0.01)
                    forces[i, 1] += rng.normal(0, 0.01)
                    forces[j, 0] -= rng.normal(0, 0.01)
                    forces[j, 1] -= rng.normal(0, 0.01)
                else:
                    # Force magnitude proportional to 1/d^2 (from U=1/d) or 1/d^3 (from U=1/d^2)
                    # Let's use U = 1/d^4 for stronger short-range repulsion
                    # F = 4/d^5
                    # This might be too strong. Let's use F = 1/d^2
                    f_mag = 1.0 / (d2 * d) # This is 1/d^3, corresponding to U = 1/d^2
                    # Scale factor
                    f_scale = 0.001 
                    fx = f_mag * f_scale * dx
                    fy = f_mag * f_scale * dy
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Boundary repulsion
        # If x < -0.5, push right. If x > 0.5, push left.
        # Let's define a "soft" boundary at 0.5.
        # Force = k * (0.5 - x) if x < 0.5, else 0? No, that pushes everything to center.
        # We want to push OUT if they go out, but they can't go out.
        # Actually, we want to maximize the distance to the boundary.
        # So we should treat the boundary as a "wall" that repels if the circle gets close?
        # No, the objective is to maximize the MIN distance to the boundary.
        # So if a circle is at x=0.49, its distance to right wall is 0.01.
        # We want to increase this.
        # So we should apply a force that pushes the circle AWAY from the closest wall?
        # Yes.
        
        # For each circle, find the closest wall distance and push away from it.
        for i in range(N):
            x = pos[i, 0]
            y = pos[i, 1]
            
            # Distances to walls
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            
            # Find min distance and corresponding direction
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            # Apply force away from the closest wall
            # Force magnitude proportional to 1/min_d^2
            if min_d < 1e-6:
                f_mag = 0.01
            else:
                f_mag = 0.001 / (min_d * min_d)
            
            if min_d == d_left:
                forces[i, 0] += f_mag
            elif min_d == d_right:
                forces[i, 0] -= f_mag
            elif min_d == d_bottom:
                forces[i, 1] += f_mag
            elif min_d == d_top:
                forces[i, 1] -= f_mag
        
        # Update velocity and position
        vel += forces * dt
        vel *= damping
        
        # Add random noise (temperature)
        if T > T_min:
            vel += rng.normal(0, T, size=(N, 2))
            T *= T_decay
        
        pos += vel * dt
        
        # Hard constraints: Clip to square
        # If we clip, we should zero out the velocity component in that direction
        for i in range(N):
            if pos[i, 0] < -0.5:
                pos[i, 0] = -0.5
                vel[i, 0] = 0
            elif pos[i, 0] > 0.5:
                pos[i, 0] = 0.5
                vel[i, 0] = 0
            
            if pos[i, 1] < -0.5:
                pos[i, 1] = -0.5
                vel[i, 1] = 0
            elif pos[i, 1] > 0.5:
                pos[i, 1] = 0.5
                vel[i, 1] = 0
        
        # Check score periodically
        if np.random.rand() < 0.01: # Check every ~100 steps
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                
    return best_pos, best_r

def _grid_init(m, n):
    """Generate a grid-like initial configuration."""
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    return np.array(points[:N])

def _random_init(rng):
    """Generate a random initial configuration."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Try multiple initializations
    # 1. Grid 7x8
    inits = []
    inits.append(_grid_init(7, 8))
    inits.append(_grid_init(8, 7))
    
    # 2. Random
    for _ in range(3):
        inits.append(_random_init(rng))
        
    # 3. Hexagonal-ish
    # Generate a hex grid
    hex_points = []
    row_height = 1.0 / 10 # Approx
    col_width = 1.0 / 8
    offset = col_width / 2
    for row in range(10):
        y = -0.5 + 0.05 + row * row_height
        for col in range(8):
            if len(hex_points) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            hex_points.append([x, y])
    inits.append(np.array(hex_points[:N]))
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Allocate remaining time for this simulation
        remaining = deadline - time.time()
        sim_time = min(remaining * 0.8, 20.0) # Cap individual sim time
        
        if sim_time < 1.0:
            continue
            
        try:
            centers, r = _simulate(init, seed=42 + i, time_limit=sim_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # If we still have time, refine the best solution with a shorter, high-precision run
    if best_centers is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        try:
            centers, r = _simulate(best_centers, seed=999, time_limit=remaining)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except Exception:
            pass
            
    # Fallback
    if best_centers is None:
        best_centers = _grid_init(7, 8)
        
    # Final validity check and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def construct():
    return _construct()
