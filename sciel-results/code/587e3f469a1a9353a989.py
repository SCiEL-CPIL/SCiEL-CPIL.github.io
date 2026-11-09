# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    """
    Strategy: Topology-Guided Linear Programming with Enhanced Topology Search.
    
    1. Define a set of structured topologies (Grids, Shelves, Mixes) and random topologies.
    2. For each topology, solve a Linear Program to maximize sum(sides) 
       subject to linear constraints defined by that topology and the unit square bounds.
    3. Return the best valid solution found.
    
    This avoids local minima of gradient-based/SA methods because the LP finds the 
    global optimum for a fixed combinatorial structure.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    tol = 1e-7
    
    def is_valid(centers, sides):
        if np.any(sides <= 0):
            return False
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(sides)):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def solve_topology(topo_dirs):
        """
        Solve LP for a given topology defined by a list of directions for each pair (i,j).
        topo_dirs[k] is the direction for the k-th pair (i,j) in lexicographic order.
        Directions:
        0: i left of j
        1: i right of j
        2: i below j
        3: i above j
        """
        num_vars = 3 * n
        c = np.zeros(num_vars)
        for i in range(n):
            c[3*i + 2] = -1.0
            
        A_list = []
        b_list = []
        
        # 1. Bounds
        for i in range(n):
            # -x_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # x_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
            # -y_i + 0.5 s_i <= 0
            row = np.zeros(num_vars)
            row[3*i+1] = -1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(0.0)
            
            # y_i + 0.5 s_i <= 1
            row = np.zeros(num_vars)
            row[3*i+1] = 1.0
            row[3*i+2] = 0.5
            A_list.append(row)
            b_list.append(1.0)
            
        # 2. Non-overlap constraints based on topology
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dir = topo_dirs[idx]
                idx += 1
                row = np.zeros(num_vars)
                if dir == 0: # i left of j
                    row[3*i] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j] = -1.0
                    row[3*j+2] = 0.5
                elif dir == 1: # i right of j
                    row[3*i] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j] = 1.0
                    row[3*j+2] = -0.5
                elif dir == 2: # i below j
                    row[3*i+1] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = -1.0
                    row[3*j+2] = 0.5
                else: # dir == 3, i above j
                    row[3*i+1] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = 1.0
                    row[3*j+2] = -0.5
                    
                A_list.append(row)
                b_list.append(0.0)
                
        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 1)) # s
            
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x = res.x
                centers = np.zeros((n, 2))
                sides = np.zeros(n)
                for i in range(n):
                    centers[i, 0] = x[3*i]
                    centers[i, 1] = x[3*i+1]
                    sides[i] = x[3*i+2]
                
                if is_valid(centers, sides):
                    score = np.sum(sides)
                    return centers, sides, score
        except Exception:
            pass
            
        return None, None, -np.inf

    def evaluate_topology(topo_dirs):
        c, s, sc = solve_topology(topo_dirs)
        if c is not None and sc > best_score:
            nonlocal best_centers, best_sides, best_score
            best_centers = c
            best_sides = s
            best_score = sc

    # Generate structured topologies
    # 1. Grids: rows x cols
    # Assign squares to grid positions. For each pair, if same row, use left/right. If same col, use above/below.
    # If different row and col, we need a consistent ordering. 
    # A simple consistent ordering: 
    # If row_i < row_j, then i is below j (dir 2).
    # If row_i > row_j, then i is above j (dir 3).
    # If row_i == row_j and col_i < col_j, then i is left of j (dir 0).
    # If row_i == row_j and col_i > col_j, then i is right of j (dir 1).
    # This creates a "dominance" topology which is valid.
    
    def generate_grid_topology(rows, cols):
        # Map 10 squares to rows x cols grid
        # If rows*cols < 10, some cells have multiple squares? No, we need 10 distinct positions.
        # So we need rows*cols >= 10.
        if rows * cols < 10:
            return None
        
        # Assign squares to positions in row-major order
        pos = []
        for r in range(rows):
            for c in range(cols):
                pos.append((r, c))
        # Take first 10
        pos = pos[:10]
        
        # Ensure we have exactly 10 positions
        # If we have more, we just use the first 10.
        
        topo = []
        for i in range(10):
            for j in range(i+1, 10):
                r_i, c_i = pos[i]
                r_j, c_j = pos[j]
                if r_i < r_j:
                    topo.append(2) # i below j
                elif r_i > r_j:
                    topo.append(3) # i above j
                else: # same row
                    if c_i < c_j:
                        topo.append(0) # i left of j
                    else:
                        topo.append(1) # i right of j
        return topo

    # Try various grid structures
    grid_configs = [
        (3, 4), # 3 rows, 4 cols
        (4, 3),
        (2, 5),
        (5, 2),
        (1, 10),
        (10, 1),
        (3, 3), # This is 9 cells, so it will return None. Need to handle 10th square specially or use 3x4.
        (4, 4),
    ]
    
    # Also try "Shelf" topologies
    # Shelves: Horizontal strips.
    # Each square is on a shelf. Squares on the same shelf are left/right.
    # Squares on different shelves are above/below.
    # This is equivalent to a grid with 1 column per shelf? No, shelves can have multiple squares.
    # This is the same as the grid topology where rows are shelves.
    
    # Try random topologies with bias towards "local" structures?
    # Just try random ones.
    
    # 1. Structured Topologies
    for r, c in grid_configs:
        if time.time() > deadline - 20: break
        topo = generate_grid_topology(r, c)
        if topo is not None:
            evaluate_topology(topo)
    
    # 2. Random Topologies
    # To improve hit rate, we can try to "repair" random topologies or use a heuristic.
    # But random LPs are fast if infeasible.
    num_random = 500
    for _ in range(num_random):
        if time.time() > deadline - 20: break
        # Random directions
        topo = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        evaluate_topology(topo)
        
        # Also try "almost grid" topologies
        # Start with a grid and perturb a few directions
        if _ % 10 == 0:
            base_topo = generate_grid_topology(3, 4)
            if base_topo is not None:
                perturbed = base_topo.copy()
                for _ in range(3):
                    idx = random.randint(0, len(perturbed)-1)
                    perturbed[idx] = random.randint(0, 3)
                evaluate_topology(perturbed)

    # Fallback to 3x3 grid if no LP solution found (should not happen)
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        # The 10th square in a 4x4 grid? No, 3x3 is 9. 4x4 is 16.
        # The fallback above uses k=4 for n=10? sqrt(10)~3.16 -> k=4.
        # So it's a 4x4 grid with 10 squares.
        # This is valid.

    return best_centers, best_sides

# EVOLVE-BLOCK-END
