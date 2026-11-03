"""
Strategy: Topology-Aware Bottleneck Relaxation.
Instead of optimizing a smooth approximation of the minimum distance over all points
(which spreads force thinly), this strategy identifies the specific "active" set of
constraints (closest pairs and closest walls) that define the current radius.
It then performs a targeted force-directed relaxation on ONLY these active points,
pushing them apart and away from the walls. This mimics the physical intuition that
to increase the radius, one must relieve the specific bottlenecks, not push everything.
A simulated annealing-style schedule is used to escape local minima of the bottleneck
configuration.
"""
import numpy as np
import time
import math
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius_and_bottlenecks(centers):
        """
        Computes the exact maximum radius and identifies the bottleneck indices.
        Returns:
        r: the radius
        active_pairs: list of (i, j) indices for pairs achieving min distance
        active_walls: list of (i, wall_idx) where wall_idx 0=x-, 1=x+, 2=y-, 3=y+
        """
        if len(centers) < 2:
            return 0.5, [], []
        
        # Wall distances
        # d_x_min = x - (-0.5) = x + 0.5
        # d_x_max = 0.5 - x
        # d_y_min = y + 0.5
        # d_y_max = 0.5 - y
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        # Pairwise distances
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        min_wall_dist = np.min([np.min(dx_min), np.min(dx_max), np.min(dy_min), np.min(dy_max)])
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        active_pairs = []
        active_walls = []
        
        # Tolerance for "active"
        tol = 1e-8
        
        # Find active pairs
        if min_pair_dist / 2.0 <= r + tol:
            # Get indices where distance is close to min
            i_idx, j_idx = np.triu_indices(N, k=1)
            d_pairs = dist_matrix[i_idx, j_idx]
            mask = d_pairs <= min_pair_dist + tol
            if np.any(mask):
                active_pairs = list(zip(i_idx[mask], j_idx[mask]))
                
        # Find active walls
        if min_wall_dist <= r + tol:
            # Check each wall
            if np.min(dx_min) <= r + tol:
                idxs = np.where(dx_min <= r + tol)[0]
                active_walls.extend([(i, 0) for i in idxs])
            if np.min(dx_max) <= r + tol:
                idxs = np.where(dx_max <= r + tol)[0]
                active_walls.extend([(i, 1) for i in idxs])
            if np.min(dy_min) <= r + tol:
                idxs = np.where(dy_min <= r + tol)[0]
                active_walls.extend([(i, 2) for i in idxs])
            if np.min(dy_max) <= r + tol:
                idxs = np.where(dy_max <= r + tol)[0]
                active_walls.extend([(i, 3) for i in idxs])

        return r, active_pairs, active_walls

    def initialize_hex_grid():
        """
        Generates a high-quality initial hexagonal packing.
        """
        # Target density: 200 circles in area 1.
        # Hex packing density is pi/sqrt(12) ~ 0.9069.
        # Area per circle ~ 1/200.
        # d^2 * sqrt(3)/2 ~ 1/200 => d ~ sqrt(2/(200*sqrt(3)))
        d = math.sqrt(2.0 / (N * math.sqrt(3)))
        
        pts = []
        row = 0
        # Generate a grid larger than the square, then crop
        # Start y from -0.5 to 0.5
        y = -0.5
        while y <= 0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                # Check if strictly inside (with small margin)
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
                x += d
            y += d * math.sqrt(3) / 2.0
            row += 1
            
        pts = np.array(pts)
        
        if len(pts) < N:
            # Fallback to square grid if hex fails (unlikely)
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # If we have more than N points, select the ones that form the best packing.
        # Simple heuristic: sort by distance to center, take N closest.
        # This ensures the "hole" in the middle is filled, maintaining symmetry.
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def relax_bottlenecks(centers, active_pairs, active_walls, step_size, temperature):
        """
        Performs a single relaxation step targeting only the active constraints.
        Returns new centers and whether a move was accepted (for SA tracking).
        """
        new_centers = centers.copy()
        
        # Compute forces on active points
        # Force magnitude is proportional to step_size
        # We want to increase the minimum distance.
        
        # 1. Pair forces: push apart
        for i, j in active_pairs:
            vec = new_centers[i] - new_centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                # Random direction if coincident
                angle = np.random.uniform(0, 2 * np.pi)
                vec = np.array([math.cos(angle), math.sin(angle)])
                dist = 1.0
            else:
                vec = vec / dist
            
            # Push i away from j, j away from i
            force = step_size * vec
            new_centers[i] += force
            new_centers[j] -= force
            
        # 2. Wall forces: push inward
        for i, wall_idx in active_walls:
            force = np.zeros(2)
            if wall_idx == 0: # x = -0.5, push +x
                force[0] = step_size
            elif wall_idx == 1: # x = 0.5, push -x
                force[0] = -step_size
            elif wall_idx == 2: # y = -0.5, push +y
                force[1] = step_size
            elif wall_idx == 3: # y = 0.5, push -y
                force[1] = -step_size
                
            new_centers[i] += force
            
        # 3. Add some noise to escape local minima (Simulated Annealing style)
        # Apply noise to ALL points, scaled by temperature
        # But keep it small so we don't destroy the packing
        noise_scale = temperature * 0.01
        if noise_scale > 0:
            new_centers += np.random.normal(0, noise_scale, size=new_centers.shape)
            
        # 4. Project back into square if necessary (hard constraint)
        # If a point goes outside, clamp it. This might violate non-overlap,
        # but the next iteration will fix it.
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        return new_centers

    # --- Main Optimization Loop ---
    
    # 1. Initialization
    centers = initialize_hex_grid()
    best_centers = centers.copy()
    best_r, _, _ = compute_radius_and_bottlenecks(centers)
    
    # 2. Optimization Parameters
    # We use a cooling schedule for the noise and step size.
    # Start with larger steps to rearrange, then smaller to polish.
    start_step = 0.05
    end_step = 0.001
    start_temp = 0.05
    end_temp = 0.0001
    
    total_iterations = 3000 # Adjust based on time budget
    # Estimate time per iteration:
    # compute_radius: O(N^2) ~ 40k ops
    # relax: O(active) ~ small
    # Should be fast.
    
    # Let's run iterations until time limit
    current_step = start_step
    current_temp = start_temp
    
    # We will use a simple "best improvement" or "first improvement" local search
    # with occasional random restarts if stuck.
    
    stuck_counter = 0
    max_stuck = 100
    
    for iter in range(total_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Cool down
        progress = iter / total_iterations
        current_step = start_step * (end_step / start_step) ** progress
        current_temp = start_temp * (end_temp / start_temp) ** progress
        
        # Get current state
        current_r, active_pairs, active_walls = compute_radius_and_bottlenecks(centers)
        
        # If no active constraints found (shouldn't happen if r is valid), break
        if not active_pairs and not active_walls:
            break
            
        # Perform relaxation
        new_centers = relax_bottlenecks(centers, active_pairs, active_walls, current_step, current_temp)
        
        # Evaluate new state
        new_r, new_pairs, new_walls = compute_radius_and_bottlenecks(new_centers)
        
        # Accept if better, or with probability exp(-delta/T) if worse (SA)
        if new_r >= current_r:
            centers = new_centers
            stuck_counter = 0
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Simulated Annealing acceptance
            delta = current_r - new_r
            if delta < 1e-12:
                # Accept neutral moves to explore
                if np.random.random() < 0.5:
                    centers = new_centers
                    stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Accept with probability
                prob = math.exp(-delta / (current_temp * 0.01 + 1e-9))
                if np.random.random() < prob:
                    centers = new_centers
                    stuck_counter += 1
                else:
                    stuck_counter += 1
                    
        # If stuck for too long, try a random perturbation to escape local minimum
        if stuck_counter > max_stuck:
            # Perturb a random subset of points
            num_perturb = N // 10
            indices = np.random.choice(N, num_perturb, replace=False)
            centers[indices] += np.random.normal(0, 0.01, size=(num_perturb, 2))
            centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            stuck_counter = 0
            # Reset step size slightly to allow recovery
            current_step = max(current_step * 2, 0.01)

    # Final polish: High precision local search on the best solution found
    # Use very small steps to squeeze out the last bit of radius
    centers = best_centers.copy()
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        r, pairs, walls = compute_radius_and_bottlenecks(centers)
        if not pairs and not walls:
            break
        new_centers = relax_bottlenecks(centers, pairs, walls, 0.0005, 0.0)
        new_r, _, _ = compute_radius_and_bottlenecks(new_centers)
        if new_r > r:
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            break # Converged

    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
