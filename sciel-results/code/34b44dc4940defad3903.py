"""
Strategy: Force-Directed Relaxation with Simulated Annealing and Grid-Based Neighbor Search.
Instead of optimizing a smooth surrogate for the min-distance, we model the problem as a
physical system where circles repel each other with a strong inverse-power force and are
repelled by the walls. We use a spatial grid to find neighbors in O(N) time, allowing for
thousands of relaxation steps. Simulated annealing (random kicks with temperature decay)
helps escape local minima. This avoids the derivative issues and local minima traps of
gradient-based methods on non-smooth min-functions.
"""

import numpy as np
import time

N = 200
# Target radius based on best known packing (slightly lower to allow optimization room)
# r_best ~ 0.0366. Diameter ~ 0.073.
# Area per circle ~ 0.0053. 200 * 0.0053 ~ 1.06. Close to 1.
# Let's aim for r ~ 0.036.

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initialization: Hexagonal packing
    # For N=200, a hex grid is a good start.
    # Approximate side length for hex packing of N circles in unit square:
    # Area ~ N * (sqrt(3)/2) * s^2 = 1 => s ~ sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    points = []
    # Generate a large hex grid and select N points closest to center
    # Grid spacing: horizontal s, vertical s*sqrt(3)/2
    max_range = int(1.0 / (s * 0.5)) + 2
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * (s / 2.0)
            y = j * s * np.sqrt(3.0) / 2.0
            # Keep only points inside [-0.5, 0.5]
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        # Select N points. To ensure good distribution, we can pick randomly or by distance to center.
        # Random selection from the valid grid points is robust.
        indices = np.random.choice(len(points), N, replace=False)
        centers = points[indices]
    else:
        # Fallback: random uniform
        centers = np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Add small random jitter to break symmetry
    centers += np.random.normal(0, 0.001, (N, 2))
    centers = np.clip(centers, -0.49, 0.49)

    def compute_score(c):
        """Compute the true radius score."""
        # Use cKDTree or simple numpy for N=200
        # N=200 is small enough for O(N^2) if vectorized, but grid is faster for force calc.
        # For scoring, O(N^2) vectorized is fast enough.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        min_b = np.min(np.minimum(dist_x, dist_y))
        
        return min(min_d / 2.0, min_b)

    best_score = compute_score(centers)
    best_centers = centers.copy()

    # Grid setup for neighbor search
    # Cell size should be larger than the maximum interaction distance.
    # We use a cutoff for repulsion. If d > cutoff, force is negligible.
    # Initial cutoff ~ 0.1. As r increases, cutoff can decrease.
    # But for simplicity, let's use a fixed cutoff that covers ~ 3-4 neighbors.
    # With r~0.036, diameter 0.072. Cutoff 0.15 is safe.
    
    def relax_step(centers, temp, k_repel, k_wall, dt):
        """
        Perform one step of force-directed relaxation with annealing.
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        
        # Grid construction
        # Cell size
        cell_size = 0.15
        grid_w = int(1.0 / cell_size) + 2
        grid_h = int(1.0 / cell_size) + 2
        
        # Map points to grid
        # x, y in [-0.5, 0.5] -> [0, grid_w]
        grid = {}
        for i in range(n):
            gx = int((centers[i, 0] + 0.5) / cell_size)
            gy = int((centers[i, 1] + 0.5) / cell_size)
            # Clamp to grid bounds
            gx = np.clip(gx, 0, grid_w - 1)
            gy = np.clip(gy, 0, grid_h - 1)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        cutoff_sq = cell_size * cell_size * 2.5 # ~ 0.375^2? No. 
        # Let's use a cutoff of 0.1.
        cutoff = 0.1
        cutoff_sq = cutoff * cutoff
        
        for i in range(n):
            ci = centers[i]
            fi = np.zeros(2)
            
            gx = int((ci[0] + 0.5) / cell_size)
            gy = int((ci[1] + 0.5) / cell_size)
            
            # Check 3x3 neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        for j in grid[key]:
                            if i == j:
                                continue
                            cj = centers[j]
                            delta = ci - cj
                            d_sq = delta[0]**2 + delta[1]**2
                            if d_sq < 1e-12:
                                # Coincident points, push randomly
                                delta = np.random.normal(0, 0.001, 2)
                                d_sq = np.dot(delta, delta)
                                if d_sq < 1e-12:
                                    continue
                            
                            if d_sq < cutoff_sq:
                                d = np.sqrt(d_sq)
                                # Repulsive force: F = k * (1/d^2) * (delta/d) = k * delta / d^3
                                # Or simpler: F = k * delta / d_sq^1.5
                                # We want strong repulsion at short range.
                                # Let's use F = k_repel * delta / d_sq
                                # This is like 1/r force.
                                # To make it stronger, use d_sq^1.5?
                                # Let's use d_sq^1.5 for 1/r^2 force magnitude.
                                # Magnitude ~ 1/d. Vector ~ 1/d^2.
                                # F_vec = k * delta / d_sq
                                # This gives magnitude k/d.
                                # As d->0, F->inf. Good.
                                
                                f_mag = k_repel / d_sq
                                fi += f_mag * delta
                                
            # Wall repulsion
            # If x < -0.5 + r_target, push right.
            # We don't know r_target exactly, but we can use a soft boundary.
            # dist_to_wall_x = 0.5 - abs(x)
            # If dist < threshold, force = k_wall / dist^2
            # Let's use threshold 0.05.
            
            dx = ci[0]
            dy = ci[1]
            
            # Left wall
            d_left = dx + 0.5
            if d_left < 0.05:
                fi[0] += k_wall / (d_left * d_left + 1e-6)
            # Right wall
            d_right = 0.5 - dx
            if d_right < 0.05:
                fi[0] -= k_wall / (d_right * d_right + 1e-6)
                
            # Bottom wall
            d_bottom = dy + 0.5
            if d_bottom < 0.05:
                fi[1] += k_wall / (d_bottom * d_bottom + 1e-6)
            # Top wall
            d_top = 0.5 - dy
            if d_top < 0.05:
                fi[1] -= k_wall / (d_top * d_top + 1e-6)
                
            forces[i] = fi
            
        # Update positions
        # Velocity Verlet or simple Euler with damping?
        # Simple: new_pos = pos + force * dt + noise
        # Damping helps convergence.
        # Let's use: pos += force * dt * damping + noise
        # But we don't have velocity.
        # Let's just do gradient descent on the potential.
        # Potential U = sum 1/d. Force = -grad U.
        # So pos += force * dt is descent.
        
        centers += forces * dt
        centers += np.random.normal(0, temp, (n, 2))
        
        # Hard constraint: clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        return centers

    # Annealing Schedule
    # Start with high temp, decay.
    # Parameters:
    # k_repel: strength of circle-circle repulsion
    # k_wall: strength of wall repulsion
    # dt: step size
    # temp: noise level
    
    # We need to balance these.
    # If k_repel is too high, forces are huge, dt must be tiny.
    # Let's scale dt with k_repel.
    
    k_repel = 1.0
    k_wall = 1.0
    dt = 0.01
    temp_start = 0.05
    temp_end = 0.001
    decay = 0.9995 # Slow decay
    
    current_temp = temp_start
    
    # Optimization Loop
    steps = 0
    while time.time() < deadline:
        # Batch of steps
        for _ in range(100):
            centers = relax_step(centers, current_temp, k_repel, k_wall, dt)
            steps += 1
            if time.time() > deadline - 5:
                break
            
        # Evaluate
        current_score = compute_score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Decay temperature
        current_temp *= decay
        
        # If temperature is very low, we might be stuck.
        # Occasionally, do a "kick" to escape local minima.
        if steps % 1000 == 0:
            # Find worst pair (closest) and perturb them
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d < 0.05: # If too close
                idx_flat = np.argmin(dists)
                i_idx = idx_flat // N
                j_idx = idx_flat % N
                # Perturb these two and their neighbors
                # Just perturb these two strongly
                noise = np.random.normal(0, 0.02, (2, 2))
                centers[i_idx] += noise[0]
                centers[j_idx] += noise[1]
                centers = np.clip(centers, -0.5, 0.5)
                
                # Slight temperature boost
                current_temp = max(current_temp, 0.01)

    return best_centers

# EVOLVE-BLOCK-END
