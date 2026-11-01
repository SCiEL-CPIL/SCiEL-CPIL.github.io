import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # --- Geometry Setup ---
    # Triangular lattice basis for hexagonal packing
    # Unit hexagon side length s=1.
    # Width across flats = sqrt(3). Height across corners = 2.
    # Lattice vectors for a hexagonal lattice of hexagons:
    # a1 = (sqrt(3), 1.5)
    # a2 = (0, 3)
    # This creates a lattice where each hexagon has 6 nearest neighbors at distance sqrt(3).
    
    SQRT3 = np.sqrt(3.0)
    a1 = np.array([SQRT3, 1.5])
    a2 = np.array([0.0, 3.0])
    
    # Define the search grid range in lattice coordinates (i, j)
    # We need to cover the area of the outer hexagon.
    # If optimal side is ~4-5, the radius is ~4-5.
    # Lattice points (i, j) map to physical pos = i*a1 + j*a2.
    # Max i, j such that |pos| < R_max.
    R_MAX = 10.0
    GRID_RANGE = int(np.ceil(R_MAX / 1.5)) + 2
    
    # Pre-compute all valid lattice points within a bounding box
    # and their indices.
    # We will map (i, j) -> index.
    
    # Generate candidate lattice points
    i_range = np.arange(-GRID_RANGE, GRID_RANGE + 1)
    j_range = np.arange(-GRID_RANGE, GRID_RANGE + 1)
    I, J = np.meshgrid(i_range, j_range)
    
    # Physical coordinates
    X = I * a1[0] + J * a2[0]
    Y = I * a1[1] + J * a2[1]
    
    # Filter points within radius R_MAX
    dist2 = X**2 + Y**2
    valid_mask = dist2 < R_MAX**2
    valid_indices = np.where(valid_mask)
    
    # Create a lookup dictionary from (i, j) to index in the valid array
    # To make bitwise collision checking fast, we need a 2D array representation of the grid
    # where each cell corresponds to a lattice point.
    
    # The grid dimensions are (len(j_range), len(i_range))
    # Let's create a 2D array 'grid' of size (2*GRID_RANGE+1, 2*GRID_RANGE+1)
    # where grid[j_idx, i_idx] = 1 if occupied, 0 otherwise.
    
    H = len(j_range)
    W = len(i_range)
    
    # Precompute neighbor offsets in (di, dj)
    # Neighbors of (i, j) in hex lattice:
    # (i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j-1), (i-1, j+1)
    # Let's verify distances:
    # d(i+1, j) = a1. |a1| = sqrt(3 + 2.25) = sqrt(5.25) ~ 2.29. 
    # Wait, the distance between centers of touching hexagons is sqrt(3) + sqrt(3) = 2*sqrt(3)? 
    # No. Inradius r = sqrt(3)/2. Distance between centers of touching hexagons = 2*r = sqrt(3).
    # My lattice vectors:
    # a1 = (sqrt(3), 1.5). |a1| = sqrt(3 + 2.25) = sqrt(5.25) = 2.29.
    # This is NOT the touching distance.
    # The touching distance is sqrt(3) ~ 1.732.
    
    # Let's re-derive the lattice for hexagonal packing of unit hexagons.
    # Vertices of hexagon at angle 0, 60, ...
    # Center at (0,0). Vertices at (1, 0), (0.5, sqrt(3)/2), ...
    # Nearest neighbor center is at (sqrt(3), 0)? No.
    # If flat side is at x=sqrt(3)/2, next hexagon flat side at x=sqrt(3)/2 + sqrt(3) = 3sqrt(3)/2?
    # Distance between centers = 2 * inradius = sqrt(3).
    # So neighbors are at distance sqrt(3).
    # Let's use basis vectors of length sqrt(3) at 60 degrees.
    # b1 = (sqrt(3), 0)
    # b2 = (sqrt(3)/2, 3/2)
    # b3 = (-sqrt(3)/2, 3/2)
    # etc.
    
    b1 = np.array([SQRT3, 0.0])
    b2 = np.array([SQRT3/2.0, 1.5])
    
    # Lattice point P = i*b1 + j*b2
    # i, j are integers.
    # Let's re-generate the grid with this basis.
    
    # R_MAX should be large enough to contain the optimal packing.
    # Score 9.05 implies side length ~ 4.5. Radius ~ 4.5.
    # R_MAX = 8.0 is safe.
    
    R_MAX = 8.0
    # Max index estimate: R_MAX / min(|b1|, |b2|) = 8 / 1.732 ~ 4.6.
    # So range -6 to 6 is sufficient.
    GRID_RANGE = 6
    
    i_vals = np.arange(-GRID_RANGE, GRID_RANGE + 1)
    j_vals = np.arange(-GRID_RANGE, GRID_RANGE + 1)
    I_grid, J_grid = np.meshgrid(i_vals, j_vals)
    
    # Physical coordinates
    X_phys = I_grid * b1[0] + J_grid * b2[0]
    Y_phys = I_grid * b1[1] + J_grid * b2[1]
    
    # Valid points within R_MAX
    valid_pts_mask = (X_phys**2 + Y_phys**2) < R_MAX**2
    
    # We will represent the state as a 2D boolean array of shape (len(j_vals), len(i_vals))
    # state[j_idx, i_idx] = True if occupied.
    
    # Precompute neighbor shifts for bitwise collision check
    # Neighbors in (di, dj):
    # (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
    # Note: b1 + b2 = (1.5*sqrt3, 1.5). Distance = sqrt(6.75 + 2.25) = 3. 
    # b1 - b2 = (0.5*sqrt3, -1.5). Distance = sqrt(0.75 + 2.25) = sqrt(3). Correct.
    # So neighbors are:
    # (1, 0) -> b1
    # (-1, 0) -> -b1
    # (0, 1) -> b2
    # (0, -1) -> -b2
    # (1, -1) -> b1 - b2
    # (-1, 1) -> -b1 + b2
    
    neighbor_offsets = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    
    def check_collisions(state):
        """
        state: 2D boolean array (H, W)
        Returns: number of collisions (pairs of adjacent occupied cells)
        """
        collisions = 0
        for di, dj in neighbor_offsets:
            # Shift state by (di, dj)
            # If state[j, i] is True and state[j+dj, i+di] is True, it's a collision.
            # We can use np.roll or slicing. Slicing is faster and avoids wrap-around if we handle bounds.
            # Since we only care about valid points, and the grid is small, 
            # we can just do:
            # shifted = np.roll(state, (-dj, -di), axis=(0, 1))
            # But roll wraps around. We must mask out the edges.
            
            # Alternative: iterate over valid indices? No, vectorized is better.
            # Let's use a padded state or just check bounds.
            # Given H, W are ~13, np.roll is fine if we zero out the wrapped parts.
            
            # Actually, simpler:
            # Create a shifted view.
            # j_new = j - dj, i_new = i - di
            # We want to check if state[j, i] AND state[j+dj, i+di] are both 1.
            
            # Let's just use a loop over the 6 neighbors with slicing, which is very fast for small arrays.
            pass
            
        # Vectorized approach:
        # For each neighbor offset (di, dj), check overlap between state and state shifted by (-di, -dj).
        # To avoid wrap-around, we can pad the state with zeros.
        
        # Since N is small (12), and grid is small (13x13), we can afford a slightly less optimized check
        # if the bitwise logic is tricky. But let's try to do it right.
        
        # Pad state with 1 layer of zeros
        padded = np.pad(state, 1, mode='constant', constant_values=0)
        
        total_collisions = 0
        for di, dj in neighbor_offsets:
            # We want to compare state[j, i] with state[j+dj, i+di]
            # In padded array (offset by 1):
            # state[j, i] is at padded[j+1, i+1]
            # state[j+dj, i+di] is at padded[j+1+dj, i+1+di]
            
            # So we compare:
            # A = padded[1:-1, 1:-1]  (which is state)
            # B = padded[1+dj : H+1+dj, 1+di : W+1+di]
            
            # Let's extract the slices
            # j indices: 0 to H-1 in state -> 1 to H in padded
            # We need B such that B[j, i] corresponds to state[j+dj, i+di]
            # So B = padded[1+dj : 1+dj+H, 1+di : 1+di+W]
            
            B = padded[1+dj : 1+dj+H, 1+di : 1+di+W]
            
            # Check overlap
            overlap = np.sum(state & B)
            total_collisions += overlap
            
        # Each collision is counted twice (once for each neighbor direction).
        # We only want the number of pairs.
        return total_collisions // 2

    def get_positions(state):
        """
        state: 2D boolean array
        Returns: Nx2 array of physical coordinates
        """
        # Find indices of True in state
        # state is (H, W). j_idx, i_idx
        j_idx, i_idx = np.where(state)
        
        # Map to physical coords
        # i_val = i_idx - GRID_RANGE
        # j_val = j_idx - GRID_RANGE
        i_vals_local = i_idx - GRID_RANGE
        j_vals_local = j_idx - GRID_RANGE
        
        x = i_vals_local * b1[0] + j_vals_local * b2[0]
        y = i_vals_local * b1[1] + j_vals_local * b2[1]
        
        return np.column_stack([x, y])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        if len(positions) == 0:
            return 0.0
            
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            # Vertices of unit hexagon at position (x, y)
            # Angles 0, 60, 120, 180, 240, 300
            angles = np.radians(np.arange(0, 360, 60))
            verts[i] = np.column_stack([
                positions[i, 0] + np.cos(angles),
                positions[i, 1] + np.sin(angles)
            ])
        
        verts_flat = verts.reshape(-1, 2)
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        
        # Support function for regular hexagon aligned with axes (flat top/bottom? or pointy?)
        # The current code uses L1..L6 which correspond to normals at 30, 90, 150...
        # Let's stick to the same definition as the current program to ensure comparability.
        # L1 = x + y/sqrt3
        # L2 = 2y/sqrt3
        # L3 = -x + y/sqrt3
        # L4 = -x - y/sqrt3
        # L5 = -2y/sqrt3
        # L6 = x - y/sqrt3
        
        L1 = x + y / SQRT3
        L2 = y * 2.0 / SQRT3
        L3 = -x + y / SQRT3
        L4 = -x - y / SQRT3
        L5 = -y * 2.0 / SQRT3
        L6 = x - y / SQRT3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    # --- Initial Configuration ---
    # Start with a valid packing: e.g., a hexagonal cluster of 12 hexagons.
    # 1 in center, 6 around, 5 more?
    # Let's just pick 12 random valid points from the grid, or use a known pattern.
    # A 12-hexagon cluster:
    # Center (0,0)
    # Neighbors (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Next layer: (2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1), (2,-2), (-2,2), (1,-2), (-1,2)
    # Let's just initialize with a few random points and let SA fix it?
    # Better: Start with a known valid configuration scaled up.
    
    # Let's create a state array
    H = len(j_vals)
    W = len(i_vals)
    
    def random_valid_state():
        state = np.zeros((H, W), dtype=bool)
        attempts = 0
        while np.sum(state) < N and attempts < 1000:
            # Pick random valid index
            r = np.random.rand()
            c = np.random.rand()
            j = int(r * H)
            i = int(c * W)
            if 0 <= j < H and 0 <= i < W:
                # Check if valid point (within R_MAX)
                # We precomputed valid_pts_mask, but let's just check distance
                i_val = i - GRID_RANGE
                j_val = j - GRID_RANGE
                x = i_val * b1[0] + j_val * b2[0]
                y = i_val * b1[1] + j_val * b2[1]
                if x*x + y*y < R_MAX*R_MAX:
                    # Check collision with existing
                    state[j, i] = True
                    if check_collisions(state) == 0:
                        pass
                    else:
                        state[j, i] = False
                        attempts += 1
                        continue
        return state

    # Better initialization: Hexagonal lattice pattern
    # Place hexagons at (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Then add more from the next ring.
    init_state = np.zeros((H, W), dtype=bool)
    # Center
    init_state[GRID_RANGE, GRID_RANGE] = True
    # Ring 1
    offsets_ring1 = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    for di, dj in offsets_ring1:
        i_idx = GRID_RANGE + di
        j_idx = GRID_RANGE + dj
        if 0 <= i_idx < W and 0 <= j_idx < H:
            init_state[j_idx, i_idx] = True
            
    # Add 5 more from Ring 2 to make 12
    # Ring 2 offsets: (2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1), (2,-2), (-2,2), (1,-2), (-1,2)
    ring2_offsets = [(2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1), (2,-2), (-2,2), (1,-2), (-1,2)]
    added = 0
    for di, dj in ring2_offsets:
        if added >= 5: break
        i_idx = GRID_RANGE + di
        j_idx = GRID_RANGE + dj
        if 0 <= i_idx < W and 0 <= j_idx < H:
            # Check if this creates a collision? 
            # In a hex lattice, ring 2 points do not touch ring 1 points?
            # Distance between (1,0) and (2,0) is b1 (touching).
            # So (2,0) touches (1,0). If (1,0) is occupied, (2,0) is invalid.
            # We need to pick non-adjacent points.
            # Let's just try to place and check.
            init_state[j_idx, i_idx] = True
            if check_collisions(init_state) == 0:
                added += 1
            else:
                init_state[j_idx, i_idx] = False

    # If we didn't get 12, fill randomly
    while np.sum(init_state) < N:
        st = random_valid_state()
        if np.sum(st) >= N:
             # Combine? No, just use st if it's valid and full
             init_state = st
             break
        # If random fails, just spread out more
        # For now, if init_state is not full, it's invalid.
        # Let's force a valid spread: linear arrangement?
        # (0,0), (2,0), (4,0)...
        init_state = np.zeros((H, W), dtype=bool)
        for k in range(N):
             i_idx = GRID_RANGE + 2*k
             j_idx = GRID_RANGE
             if i_idx < W:
                 init_state[j_idx, i_idx] = True
        break

    # Ensure init_state is valid
    if check_collisions(init_state) > 0:
        # Fallback: Linear
        init_state = np.zeros((H, W), dtype=bool)
        for k in range(N):
             i_idx = GRID_RANGE + 2*k
             j_idx = GRID_RANGE
             if i_idx < W:
                 init_state[j_idx, i_idx] = True

    best_state = init_state.copy()
    best_positions = get_positions(best_state)
    best_side = required_outer_side(best_positions)
    
    # --- Simulated Annealing ---
    T_start = 5.0
    T_end = 0.01
    max_iters = 20000
    
    current_state = init_state.copy()
    current_positions = get_positions(current_state)
    current_side = required_outer_side(current_positions)
    
    # Objective: Side length + Penalty * Collisions
    # We want to minimize Side length subject to Collisions = 0.
    # If Collisions > 0, we are in an invalid state.
    # Penalty should be high enough to prevent collisions in the final state,
    # but allow them during search to escape local minima.
    
    penalty_weight = 10.0
    
    def get_objective(state):
        cols = check_collisions(state)
        if cols > 0:
            # Penalize heavily
            pos = get_positions(state)
            side = required_outer_side(pos)
            return side + penalty_weight * cols
        else:
            pos = get_positions(state)
            return required_outer_side(pos)

    current_obj = get_objective(current_state)
    best_obj = current_obj
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Generate neighbor state
        # Move one hexagon to a random nearby lattice point
        # Pick a random occupied cell
        occupied_indices = np.where(current_state)
        if len(occupied_indices[0]) == 0:
            break
        k = np.random.randint(len(occupied_indices[0]))
        j_old = occupied_indices[0][k]
        i_old = occupied_indices[1][k]
        
        # Pick a random new location within a small radius
        # Radius 1 or 2 in lattice steps
        radius = 2
        di = np.random.randint(-radius, radius + 1)
        dj = np.random.randint(-radius, radius + 1)
        i_new = i_old + di
        j_new = j_old + dj
        
        if i_new < 0 or i_new >= W or j_new < 0 or j_new >= H:
            continue
            
        # Don't move to same spot
        if i_new == i_old and j_new == j_old:
            continue
            
        # Create new state
        new_state = current_state.copy()
        new_state[j_old, i_old] = False
        new_state[j_new, i_new] = True
        
        # Check if new state has same number of particles (should, unless we moved out of bounds which we checked)
        # Check collisions
        new_obj = get_objective(new_state)
        
        delta = new_obj - current_obj
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_state = new_state
            current_obj = new_obj
            current_positions = get_positions(current_state)
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_state = current_state.copy()
                best_positions = current_positions.copy()
                
                # If we found a valid state (0 collisions), update best_side
                if check_collisions(best_state) == 0:
                    best_side = required_outer_side(best_positions)

    # Final validation
    if check_collisions(best_state) > 0:
        # Try to repair: if still invalid, use the initial valid state
        best_state = init_state
        best_positions = get_positions(best_state)
        best_side = required_outer_side(best_positions)
        
    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    # The outer hexagon is centered at (0,0) and has side length best_side.
    # Rotation is 0.
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
