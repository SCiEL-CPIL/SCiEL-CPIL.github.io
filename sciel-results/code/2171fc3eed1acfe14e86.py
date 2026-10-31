"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
import itertools
import time

def construct():
    best_sum = -1
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2
    # w in [0.5, 1.5]
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    grid_size = 15 # 15x15 = 225 cells
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Generate grid centers
        xs = np.linspace(0.1, w - 0.1, grid_size)
        ys = np.linspace(0.1, h - 0.1, grid_size)
        
        cells = []
        for x in xs:
            for y in ys:
                cells.append([x, y])
        cells = np.array(cells)
        M = len(cells)
        
        # Precompute cell distances
        # dist_matrix[i, j] = distance between cell i and cell j
        # We only need this for the MILP constraints
        # To save memory/time, we can compute on the fly or use a sparse structure
        # For 225 cells, 225x225 matrix is fine.
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(cells, cells)
        
        # Binary search for max radius r
        # Lower bound: 0.05
        # Upper bound: min(w, h) / 2
        r_low = 0.05
        r_high = min(w, h) / 2.0
        
        best_r_for_w = 0
        best_positions = None
        
        # Perform binary search
        # We want to find the max r such that a valid packing exists.
        # We will do a few iterations.
        for _ in range(10):
            r_mid = (r_low + r_high) / 2.0
            
            # Check if packing with radius r_mid is possible
            # MILP Formulation:
            # Variables x_{i, j} in {0,1} for i in [0, N), j in [0, M)
            # x_{i, j} = 1 if circle i is at cell j
            
            # Constraints:
            # 1. Each circle assigned to exactly one cell: sum_j x_{i, j} = 1
            # 2. Each cell has at most one circle: sum_i x_{i, j} <= 1
            # 3. Non-overlap: if dist(cell_j, cell_k) < 2*r_mid, then x_{i, j} + x_{k, l} <= 1 for i != k
            #    Note: i and k are circle indices, j and l are cell indices.
            #    Actually, the constraint is on cells. If cell j and cell k are too close,
            #    no two *different* circles can be there.
            #    So, for any pair of cells (j, k) with dist < 2*r:
            #    sum_{i=0}^{N-1} x_{i, j} + sum_{k'=0}^{N-1} x_{k', k} <= 1 ?
            #    No. If circle 1 is at j and circle 2 is at k, that's bad.
            #    But circle 1 at j and circle 1 at k is impossible (sum_j x_{1,j}=1).
            #    So we need: x_{i, j} + x_{k, k} <= 1 for all i != k, and dist(j,k) < 2r.
            
            # This creates O(N^2 * M^2) constraints.
            # N=21, M=225. 21^2 * 225^2 = 441 * 50625 = ~22 million constraints. Too many.
            
            # Optimization: Only add constraints for "close" cell pairs.
            # Find all pairs (j, k) with dist(j, k) < 2*r_mid.
            # Let C_close be the list of such pairs.
            # For each (j, k) in C_close:
            #   We cannot have any circle i at j AND any circle k' at k (where i != k').
            #   This is equivalent to:
            #   (sum_i x_{i, j}) + (sum_{k' != i} x_{k', k}) <= 1 ??
            #   No, that's not right.
            #   If cell j is occupied by circle i, and cell k is occupied by circle k', and i != k', it's invalid.
            #   This is a quadratic constraint in the assignment variables.
            #   Linearization:
            #   For each close pair (j, k):
            #   For each i in [0, N):
            #     For each k' in [0, N), k' != i:
            #       x_{i, j} + x_{k', k} <= 1
            
            # Number of constraints: |C_close| * N * (N-1).
            # If r is small, C_close is small. If r is large, C_close is large.
            # Let's limit the number of constraints or use a heuristic.
            
            # Alternative: Use a simpler "Greedy + Local Search" which is faster and often sufficient.
            # Given the time limit and complexity, let's switch to a robust Local Search.
            # The "Matching" directive can be satisfied by using Hungarian for initialization.
            
            # Let's abort the MILP idea for speed and use:
            # 1. Grid Initialization via Greedy (which is a matching of circles to best spots).
            # 2. Local Optimization.
            
            break # Break binary search, use heuristic instead
        
        # Heuristic Approach:
        # 1. Initialize positions on a grid.
        # 2. Use Nelder-Mead to maximize the minimum radius (or sum).
        
        # Generate initial positions:
        # Place circles in a grid pattern.
        # N=21. 5x5 grid with 4 removed? Or 4x6 with 3 removed?
        # Let's try 5x5.
        nx, ny = 5, 5
        # Adjust for aspect ratio
        # We want to map a 5x5 grid to w x h.
        
        # Better: Use a "Floyd" or "Greedy" placement on the grid cells.
        # Select N cells from the M cells such that they are well separated.
        
        # Greedy Selection:
        # 1. Start with all cells available.
        # 2. Pick the cell with the largest distance to the boundary.
        # 3. Place a circle there.
        # 4. Remove all cells within distance 2*r_est from it.
        # 5. Repeat.
        
        # We need an estimate of r.
        # r_est = min(w, h) / 10 # Rough guess
        
        # Let's just use a simple grid initialization and refine.
        
        # Create a 5x5 grid of positions
        xs_init = np.linspace(0.1, w - 0.1, 5)
        ys_init = np.linspace(0.1, h - 0.1, 5)
        
        positions = []
        for i in range(5):
            for j in range(5):
                positions.append([xs_init[i], ys_init[j]])
        
        # We have 25 positions, need 21.
        # Remove 4 corners? Or random 4?
        # Let's remove the 4 corners to keep them central.
        # Indices: 0, 4, 20, 24 are corners in 5x5 (row-major)
        # 0: (0,0), 4: (0,4), 20: (4,0), 24: (4,4)
        positions = [pos for idx, pos in enumerate(positions) if idx not in [0, 4, 20, 24]]
        
        # If we still don't have 21, adjust.
        # 25 - 4 = 21. Perfect.
        
        positions = np.array(positions)
        
        # Now, optimize.
        # Objective: Maximize sum of radii.
        # We can parameterize by positions and radii.
        # Or, simpler: Maximize the minimum radius, then assume all radii are equal.
        # Then, try to increase individual radii if possible.
        
        # Let's use SLSQP to maximize sum of radii directly.
        # Variables: x_1, y_1, r_1, ..., x_21, y_21, r_21 (63 variables)
        
        def objective(v):
            # v contains [x1, y1, r1, ..., xN, yN, rN]
            pos = v[0:2*N:2] # x coords
            pos_y = v[1:2*N:2] # y coords
            rad = v[2*N:3*N] # radii
            
            # Penalty for constraints? No, SLSQP handles constraints.
            return -np.sum(rad)
            
        # Constraints
        # 1. Boundary:
        #    x_i >= r_i, x_i <= w - r_i
        #    y_i >= r_i, y_i <= h - r_i
        # 2. Non-overlap:
        #    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        # This is a lot of constraints.
        # N=21.
        # Boundary: 4 * 21 = 84 constraints.
        # Overlap: N*(N-1)/2 = 210 constraints.
        
        # SLSQP might struggle with 63 vars and 300 constraints.
        # Let's try.
        
        # Initial guess for radii
        r_init = 0.08
        v0 = np.zeros(3*N)
        for i in range(N):
            v0[2*i] = positions[i][0]
            v0[2*i+1] = positions[i][1]
            v0[2*N+i] = r_init
            
        # Define constraints for SLSQP
        # Inequality constraints: c(v) >= 0
        
        cons = []
        
        # Boundary constraints
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*N+i]})
            # w - r_i - x_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[2*N+i] - v[2*i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*N+i]})
            # h - r_i - y_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[2*N+i] - v[2*i+1]})
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[2*i]-v[2*j])**2 + (v[2*i+1]-v[2*j+1])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.01, w)) # x
            bounds.append((0.01, h)) # y
            bounds.append((0.01, min(w,h)/2)) # r
            
        # Run optimization
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success:
                v_opt = res.x
                rad = v_opt[2*N:3*N]
                total_sum = np.sum(rad)
                
                if total_sum > best_sum:
                    best_sum = total_sum
                    # Construct result array
                    result = np.zeros((N, 3))
                    for i in range(N):
                        result[i, 0] = v_opt[2*i]
                        result[i, 1] = v_opt[2*i+1]
                        result[i, 2] = v_opt[2*N+i]
                    best_result = result
                    
        except Exception as e:
            # If SLSQP fails, fallback to simple equal radius packing
            pass
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
