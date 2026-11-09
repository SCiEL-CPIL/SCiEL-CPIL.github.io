# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

N = 26

def _init_grid_hex(n):
    """
    Initialize circles in a hexagonal-like grid pattern.
    This provides a good starting point for packing.
    """
    positions = []
    # We want roughly 26 circles.
    # A 5x5 grid is 25. Let's add 1 more or use a denser pattern.
    # Hex grid:
    # Row 0: 5 circles
    # Row 1: 5 circles (offset)
    # Row 2: 5 circles
    # Row 3: 5 circles (offset)
    # Row 4: 6 circles?
    
    # Let's try a 5x6 rectangular grid with hex offsets and pick 26 best.
    # Or simpler: 5x5 grid + 1 center.
    
    # Better: Generate a 6x5 grid of points, jitter them, and select 26.
    # Actually, for N=26, a 5x5 grid is 25. 
    # Let's use a 5x5 grid and place the 26th in the center of the square?
    # No, that creates overlap.
    
    # Let's use a 6x5 grid (30 points) and randomly drop 4.
    rows = 5
    cols = 6
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows
            # Hex offset for odd rows
            if r % 2 == 1:
                x += 0.5 / cols
            # Clip to ensure inside
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            pts.append([x, y])
            
    # Randomly select 26
    indices = random.sample(range(len(pts)), n)
    return np.array([pts[i] for i in indices])

def _init_random(n):
    """
    Random initialization.
    """
    return np.random.uniform(0.1, 0.9, (n, 2))

def _init_grid_square(n):
    """
    Square grid initialization.
    """
    side = int(math.ceil(math.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    return np.array(positions[:n])

def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 with disjoint interiors,
    maximizing the sum of their radii.
    
    Uses a force-directed relaxation method with iterative radius growth.
    """
    
    best_solution = None
    best_sum = -1.0
    
    n_restarts = 20
    max_iterations = 5000
    growth_rate = 0.05
    force_strength = 1.0
    step_size = 0.1
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for overhead
    
    initializers = [_init_grid_hex, _init_random, _init_grid_square]
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Choose initializer
        init_func = initializers[restart % len(initializers)]
        positions = init_func(N).copy()
        radii = np.full(N, 0.01)
        
        # Relaxation loop
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Calculate pairwise distances
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist = np.sqrt(dist_sq)
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            dist[~mask] = dist[~mask] # No change, just ensuring shape
            # Actually, we need to be careful with division by zero later.
            
            # 2. Calculate overlaps
            # Overlap exists if dist[i,j] < radii[i] + radii[j]
            sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlap = sum_radii - dist
            # Only consider overlaps where mask is True
            overlap[~mask] = 0
            overlap = np.maximum(overlap, 0)
            
            # Check if system is settled (no overlaps)
            if np.max(overlap) < 1e-6:
                # Try to grow radii
                # Calculate max possible radius for each circle
                # Boundary constraints
                max_r_boundary = np.minimum(
                    np.minimum(positions[:, 0], 1 - positions[:, 0]),
                    np.minimum(positions[:, 1], 1 - positions[:, 1])
                )
                
                # Neighbor constraints
                # For circle i, max r_i such that r_i + r_j <= dist[i,j] for all j
                # r_i <= dist[i,j] - r_j
                # So max_r_i = min_j (dist[i,j] - r_j)
                
                # term[i,j] = dist[i,j] - radii[j]
                term = dist - radii[np.newaxis, :]
                np.fill_diagonal(term, np.inf)
                max_r_inter = np.min(term, axis=1)
                
                max_r = np.minimum(max_r_boundary, max_r_inter)
                
                # Grow radii
                # Only grow if there is room
                growth = np.maximum(max_r - radii, 0)
                # Damp growth to prevent instability
                new_radii = radii + growth * growth_rate
                
                # Check if radii changed significantly
                if np.max(np.abs(new_radii - radii)) < 1e-7:
                    # Converged
                    radii = new_radii
                    break
                
                radii = new_radii
                continue # Skip force application if no overlaps, just grow
            
            # 3. Apply forces to resolve overlaps
            # Force on i from j is proportional to overlap, direction is unit vector from j to i
            dist_safe = np.where(dist > 1e-8, dist, 1e-8)
            unit_vec = diff / dist_safe[:, :, np.newaxis]
            
            # F_i = sum_j (overlap_ij * unit_vec_ij)
            # Note: diff[i,j] = pos[i] - pos[j]. So unit_vec points from j to i.
            # If they overlap, we want to push i away from j. So force direction is correct.
            forces = np.sum(overlap[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Boundary forces
            # Push back into square if outside
            # Left
            viol_left = np.maximum(radii - positions[:, 0], 0)
            forces[:, 0] += force_strength * viol_left
            # Right
            viol_right = np.maximum(positions[:, 0] - (1 - radii), 0)
            forces[:, 0] -= force_strength * viol_right
            # Bottom
            viol_bottom = np.maximum(radii - positions[:, 1], 0)
            forces[:, 1] += force_strength * viol_bottom
            # Top
            viol_top = np.maximum(positions[:, 1] - (1 - radii), 0)
            forces[:, 1] -= force_strength * viol_top
            
            # Update positions
            # Use a step size that scales with force magnitude?
            # Fixed step is simpler but might be unstable.
            # Let's use a constant step size for now.
            positions += forces * step_size
            
            # Clamp to boundaries
            positions[:, 0] = np.clip(positions[:, 0], 0.0, 1.0)
            positions[:, 1] = np.clip(positions[:, 1], 0.0, 1.0)
            
            # Optional: Small jitter to escape local minima
            # if iteration % 100 == 0:
            #     positions += np.random.normal(0, 0.001, positions.shape)
            #     positions[:, 0] = np.clip(positions[:, 0], 0.0, 1.0)
            #     positions[:, 1] = np.clip(positions[:, 1], 0.0, 1.0)
            
        # Final Validation and Cleanup
        # We need to ensure strict non-overlap and boundary containment.
        # The loop above might leave tiny overlaps or boundary violations.
        
        # Iterative shrinking to ensure validity
        valid = False
        for _ in range(50):
            valid = True
            
            # Check boundaries
            for i in range(N):
                # If circle sticks out, shrink radius to fit
                r_max_boundary = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
                if radii[i] > r_max_boundary:
                    radii[i] = r_max_boundary
                    valid = False
            
            # Check pairwise overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = radii[i] + radii[j]
                    if dist_sq < min_dist * min_dist:
                        # Overlap
                        dist = math.sqrt(dist_sq)
                        overlap = min_dist - dist
                        # Shrink both radii by half the overlap
                        shrink = overlap / 2.0
                        radii[i] = max(radii[i] - shrink, 0)
                        radii[j] = max(radii[j] - shrink, 0)
                        valid = False
            
            if valid:
                break
        
        if valid:
            total = np.sum(radii)
            if total > best_sum:
                best_sum = total
                best_solution = np.column_stack([positions[:, 0], positions[:, 1], radii])
    
    # Fallback
    if best_solution is None:
        # Simple grid fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        if len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
