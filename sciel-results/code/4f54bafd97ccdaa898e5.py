"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Discrete Grid Parameters
    # We use a grid to discretize the rectangle.
    # Perimeter 4 => W+H=2.
    # Let's fix a grid resolution. Higher resolution = better precision but slower.
    # 100x100 grid means cell size 0.02 (if W=1, H=1).
    # Let's adapt grid to aspect ratio.
    
    best_score = 0.0
    best_result = None
    
    # Aspect ratios to try
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 0.6]
    
    for W in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        H = 2.0 - W
        if H <= 0.1 or W <= 0.1:
            continue
        
        # Grid dimensions
        # We want roughly square cells.
        # Let's pick a grid width such that cell size is ~0.01-0.02
        # If W=1, grid_w=100 -> cell=0.01
        # If W=1.5, grid_w=150 -> cell=0.01
        # Let's fix grid_w = 100 for W~1, scale linearly?
        # Actually, let's just use a fixed number of cells per unit length.
        # Resolution: 50 cells per unit length.
        res = 50
        gw = int(W * res)
        gh = int(H * res)
        
        if gw < 10 or gh < 10:
            continue
            
        # Total cells
        total_cells = gw * gh
        
        # State: 
        # occupancy: boolean array of shape (gw, gh)
        # radii: array of length N
        # positions: array of (N, 2) storing (col, row) in grid coords
        
        # Initialize
        # Randomly place N circles in distinct cells
        # Initial radius: small
        r_min = 1.0 / res # 1 cell width
        
        def random_init():
            cells = np.random.choice(total_cells, size=N, replace=False)
            cols = cells % gw
            rows = cells // gw
            pos = np.column_stack((cols, rows))
            # Initial radii: 1.5 cells
            radii = np.full(N, 1.5)
            return pos, radii
            
        def get_config(pos, radii):
            """Convert grid positions to continuous coordinates and check validity/score."""
            # Grid center of cell (c, r) is at (c + 0.5) / res * W? 
            # No, let's map grid indices to [0, W] and [0, H].
            # Cell (c, r) covers [c/res*W, (c+1)/res*W] ?
            # Let's define cell size dx = W/gw, dy = H/gh.
            # Center of cell (c, r) is (c+0.5)*dx, (r+0.5)*dy.
            
            dx = W / gw
            dy = H / gh
            
            xs = (pos[:, 0] + 0.5) * dx
            ys = (pos[:, 1] + 0.5) * dy
            
            # Calculate max possible radius for each circle based on neighbors and walls
            # In this discrete model, we assume circles can grow until they hit a neighbor or wall.
            # However, for the "Lattice Gas" model, usually the radius is fixed or determined by the grid.
            # The prompt suggests: "radii dynamically calculated as the minimum distance to grid boundaries or neighbors".
            # This implies we can pack tighter by allowing circles to be different sizes?
            # But if they are on a grid, their centers are fixed.
            # The radius of circle i is min(
            #   dist to left wall, dist to right wall, dist to bottom wall, dist to top wall,
            #   min_{j!=i} (dist(i,j) - r_j)  # This is coupled!
            # )
            # This coupling makes it hard to calculate independently.
            # Alternative interpretation: 
            # We fix the centers on the grid. We want to maximize sum(r).
            # This is equivalent to finding the largest disjoint disks centered at these grid points.
            # This can be solved by binary search on a common scaling factor? No, radii can be different.
            # But usually, in such packings, equal radii are optimal or near-optimal for regular grids.
            # Let's assume for the discrete search that we seek a configuration where all radii are equal to R.
            # Then the score is N * R.
            # R is determined by the "bottleneck" distance:
            # R_max = min(
            #   min_i (dist(i, walls)),
            #   min_{i<j} (dist(i,j) / 2)
            # )
            # If we allow variable radii, it's a linear programming problem for given centers.
            # Given the time constraint, let's stick to Equal Radii for the discrete search, 
            # and then refine with continuous optimization if needed? 
            # The directive says "radii dynamically calculated".
            # Let's calculate the max uniform radius R for the given grid positions.
            
            # 1. Wall constraints
            r_wall = np.inf
            for i in range(N):
                r_wall = min(r_wall, xs[i], W - xs[i], ys[i], H - ys[i])
                
            # 2. Neighbor constraints
            r_neighbor = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    r_neighbor = min(r_neighbor, dist / 2.0)
                    
            R = min(r_wall, r_neighbor)
            
            # Check if valid (R > 0)
            if R <= 0:
                return 0.0, None
                
            score = N * R
            result = np.column_stack((xs, ys, np.full(N, R)))
            return score, result
            
        # Simulated Annealing on the Grid
        # Moves:
        # 1. Pick a random circle, move it to a random empty cell.
        # 2. Pick two random circles, swap their cells.
        
        pos, radii = random_init()
        current_score, _ = get_config(pos, radii)
        best_local_score = current_score
        best_local_pos = pos.copy()
        
        # Occupancy map for fast collision check? 
        # For N=21, O(N^2) is fine.
        
        T = 1.0
        cooling = 0.999
        steps = 0
        
        while time.time() < deadline - 5:
            steps += 1
            if steps % 1000 == 0:
                # Check validity of best local
                s, res = get_config(best_local_pos, radii)
                if s > best_local_score:
                    best_local_score = s
            
            # Generate move
            move_type = random.random()
            if move_type < 0.5:
                # Swap
                i, j = np.random.choice(N, 2, replace=False)
                new_pos = pos.copy()
                new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
            else:
                # Move
                i = np.random.choice(N)
                # Pick random cell
                new_cell = np.random.randint(0, total_cells)
                new_c, new_r = new_cell % gw, new_cell // gw
                # Check if occupied
                occupied = False
                for k in range(N):
                    if k != i and pos[k, 0] == new_c and pos[k, 1] == new_r:
                        occupied = True
                        break
                if occupied:
                    continue
                new_pos = pos.copy()
                new_pos[i] = [new_c, new_r]
                
            new_score, _ = get_config(new_pos, radii)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                pos = new_pos
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_pos = pos.copy()
                    
            T *= cooling
            if T < 0.01:
                break
        
        # Final check
        final_score, final_res = get_config(best_local_pos, radii)
        if final_score > best_score:
            best_score = final_score
            best_result = final_res

    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Simple grid fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i+0.5)/side, (j+0.5)/side, 0.5/side - 0.01])
        best_result = np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
