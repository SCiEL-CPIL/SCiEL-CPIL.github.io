"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Uses a Force-Directed Relaxation approach.
    1. Initialize circles on a jittered grid.
    2. Iteratively resolve overlaps by pushing circles apart.
    3. Grow radii when no overlaps exist.
    4. Multi-start with different jitter seeds.
    """
    
    def check_validity(pos, rad):
        """
        Check if circles are within [0,1]^2 and disjoint.
        Returns True if valid, False otherwise.
        """
        # Boundary check
        if np.any(pos[:, 0] - rad < -1e-9) or np.any(pos[:, 0] + rad > 1 + 1e-9):
            return False
        if np.any(pos[:, 1] - rad < -1e-9) or np.any(pos[:, 1] + rad > 1 + 1e-9):
            return False
        
        # Overlap check
        # Calculate squared distances between all pairs
        # We can vectorize this for speed
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx**2 + dy**2
        
        # Sum of radii
        rad_sum = rad[:, np.newaxis] + rad[np.newaxis, :]
        
        # We only care about i < j
        i, j = np.triu_indices(N, k=1)
        if np.any(dist_sq[i, j] < rad_sum[i, j]**2 - 1e-9):
            return False
            
        return True

    def force_directed_relax(pos, rad, max_iters=5000, time_limit=10.0):
        """
        Relax positions to resolve overlaps, then grow radii.
        """
        pos = pos.copy()
        rad = rad.copy()
        
        start_time = time.time()
        lr = 0.5 # Learning rate / step size for position updates
        growth_rate = 0.001 # How much to grow radii each growth phase
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Resolve Overlaps (Force Directed)
            forces = np.zeros_like(pos)
            overlap_count = 0
            
            # Calculate forces
            # Vectorized calculation of overlaps
            dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
            dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
            dist_sq = dx**2 + dy**2
            dist = np.sqrt(dist_sq + 1e-12) # Avoid div by zero
            
            rad_sum = rad[:, np.newaxis] + rad[np.newaxis, :]
            
            # Overlap depth: max(0, rad_sum - dist)
            overlap = np.maximum(0, rad_sum - dist)
            
            # Force direction: (pos_j - pos_i) / dist
            # Force magnitude: proportional to overlap
            # We want to push i away from j if they overlap.
            # Force on i from j: -overlap * (pos_i - pos_j) / dist
            
            # Mask for i != j
            mask = ~np.eye(N, dtype=bool)
            
            # Normalize direction
            dir_x = dx / dist
            dir_y = dy / dist
            
            # Force contribution to i from j is -overlap * dir
            # Sum over j for each i
            # forces[i] = sum_j (-overlap[i,j] * dir[i,j])
            
            # Only count i < j to avoid double counting and sign errors?
            # Actually, the force on i due to j is opposite to force on j due to i.
            # Let's just sum all interactions.
            
            fx = -np.sum(overlap * dir_x * mask, axis=1)
            fy = -np.sum(overlap * dir_y * mask, axis=1)
            
            forces[:, 0] = fx
            forces[:, 1] = fy
            
            overlap_count = np.sum(overlap)
            
            # Update positions
            pos += lr * forces
            
            # 2. Handle Boundaries
            # If a circle goes out of bounds, push it back
            # x < rad -> pos_x = rad
            # x + rad > 1 -> pos_x = 1 - rad
            
            # Simple projection
            pos[:, 0] = np.clip(pos[:, 0], rad, 1 - rad)
            pos[:, 1] = np.clip(pos[:, 1], rad, 1 - rad)
            
            # 3. Grow Radii
            # If no significant overlap, try to grow
            if overlap_count < 1e-6:
                # Try to grow all radii slightly
                old_rad = rad.copy()
                rad += growth_rate
                
                # Check if new radii are valid with current positions
                if not check_validity(pos, rad):
                    # Binary search for max valid growth? Or just revert?
                    # Revert for simplicity and stability
                    rad = old_rad
                else:
                    # If valid, we can also try to push circles towards center?
                    # No, let the relaxation handle it.
                    pass
            else:
                # If there is overlap, maybe reduce learning rate slightly?
                # Or just let it be.
                pass
        
        # Final cleanup: ensure strict validity
        if not check_validity(pos, rad):
             # Scale down radii until valid
             scale = 1.0
             while not check_validity(pos, rad * scale) and scale > 0.01:
                 scale *= 0.95
             rad *= scale
             
        return pos, rad

    def initial_placement(seed):
        np.random.seed(seed)
        # Use a 5x5 grid + 1 extra logic, but jittered heavily
        # Or just random uniform start? Random start is often worse for packing.
        # A jittered grid is a good heuristic.
        
        # Let's use a 5x5 grid for 25 circles, and put the 26th in a random spot
        # Or better: 6x6 grid, pick 26 best spots?
        # Let's stick to 5x5 + 1 for simplicity, but with larger jitter.
        
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # 5x5 grid
        side = 5
        for i in range(25):
            row = i // side
            col = i % side
            # Jitter
            jx = np.random.uniform(-0.1, 0.1)
            jy = np.random.uniform(-0.1, 0.1)
            positions[i, 0] = (col + 0.5) / side + jx
            positions[i, 1] = (row + 0.5) / side + jy
            radii[i] = 0.15 # Start small, will grow
            
        # 26th circle
        # Place it in a random corner-ish area or center?
        # Center is crowded. Let's put it in a random valid-ish spot.
        # Actually, let's just place it randomly and let relaxation fix it.
        positions[25, 0] = np.random.uniform(0.2, 0.8)
        positions[25, 1] = np.random.uniform(0.2, 0.8)
        radii[25] = 0.15
        
        # Clip to valid range (ensure radius fits)
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
        
        return positions, radii

    # Main logic
    start_time = time.time()
    time_limit = 108
    
    best_score = -np.inf
    best_pos = None
    best_rad = None
    
    num_runs = 10
    time_per_run = time_limit / num_runs
    
    for run in range(num_runs):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
            
        seed = 100 + run * 7919
        init_pos, init_rad = initial_placement(seed)
        
        # Allocate time for relaxation
        relax_time = min(time_per_run, remaining - 1)
        
        pos, rad = force_directed_relax(init_pos, init_rad, max_iters=20000, time_limit=relax_time)
        
        if check_validity(pos, rad):
            score = np.sum(rad)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_rad = rad.copy()
                
    # Fallback if no valid solution found (should not happen)
    if best_pos is None:
        # Return a simple grid
        side = 5
        cells = []
        for i in range(25):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.1])
        # Add 26th
        cells.append([0.5, 0.5, 0.05])
        return np.array(cells)
        
    return np.column_stack([best_pos, best_rad])
# EVOLVE-BLOCK-END
