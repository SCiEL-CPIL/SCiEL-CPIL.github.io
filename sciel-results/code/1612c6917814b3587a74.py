# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog
import itertools

def squares_in_square10() -> tuple:
    """
    Strategy: Topology-Guided Linear Programming.
    
    Instead of continuous SA with penalties, we:
    1. Generate candidate topologies (orderings of square edges in x and y).
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

    def solve_topology(x_order, y_order):
        """
        Solve LP for a given topology defined by permutations x_order, y_order.
        x_order[i] is the rank of square i's left edge? No, let's define it as 
        the order of squares by their left edge x-coordinate.
        Actually, a simpler constraint set is:
        For all i, j, if i is to the left of j in x_order, then x_i + s_i <= x_j.
        But this is too restrictive (forces full separation in x for all pairs).
        
        Better Topology Definition:
        We define the "stacking" order.
        Let's use a simpler heuristic: 
        We will define constraints based on a "grid" structure or "sweep" structure.
        
        Alternative LP Formulation:
        Variables: x_i, y_i, s_i for i in 0..n-1.
        Objective: max sum(s_i)
        Constraints:
        1. s_i > 0
        2. x_i - s_i/2 >= 0
        3. x_i + s_i/2 <= 1
        4. y_i - s_i/2 >= 0
        5. y_i + s_i/2 <= 1
        6. Non-overlap: For each pair (i,j), one of 4 conditions must hold:
           x_i + s_i <= x_j  OR x_j + s_j <= x_i  OR y_i + s_i <= y_j  OR y_j + s_j <= y_i
           
        This is an OR-constraint, which is non-linear/disjunctive.
        To make it an LP, we must fix which condition holds for each pair.
        This is the "topology".
        
        We will generate random topologies by randomly assigning a "direction" 
        for each pair (i,j): 
        0: i is left of j
        1: i is right of j
        2: i is below j
        3: i is above j
        
        Then we add the corresponding linear constraint.
        If the LP is infeasible, we discard this topology.
        """
        # Variables: x_0, y_0, s_0, x_1, y_1, s_1, ...
        # Indexing: x_i is at 3*i, y_i at 3*i+1, s_i at 3*i+2
        num_vars = 3 * n
        
        # Objective: maximize sum(s_i) -> minimize -sum(s_i)
        c = np.zeros(num_vars)
        for i in range(n):
            c[3*i + 2] = -1.0
            
        # Inequalities A_ub x <= b_ub
        A_list = []
        b_list = []
        
        # 1. Bounds
        # x_i - s_i/2 >= 0  => -x_i + 0.5*s_i <= 0
        # x_i + s_i/2 <= 1  => x_i + 0.5*s_i <= 1
        # y_i - s_i/2 >= 0  => -y_i + 0.5*s_i <= 0
        # y_i + s_i/2 <= 1  => y_i + 0.5*s_i <= 1
        # s_i >= 0 (handled by bounds)
        
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
        # For each pair (i, j) with i < j
        for i in range(n):
            for j in range(i+1, n):
                # Randomly choose direction
                # 0: i left of j  => x_i + s_i/2 <= x_j - s_j/2  => x_i + 0.5s_i - x_j + 0.5s_j <= 0
                # 1: i right of j => x_j + s_j/2 <= x_i - s_i/2  => -x_i + 0.5s_i + x_j - 0.5s_j <= 0
                # 2: i below j    => y_i + s_i/2 <= y_j - s_j/2  => y_i + 0.5s_i - y_j + 0.5s_j <= 0
                # 3: i above j    => y_j + s_j/2 <= y_i - s_i/2  => -y_i + 0.5s_i + y_j - 0.5s_j <= 0
                
                # Heuristic: If we assume a grid-like structure, we can bias the choice.
                # But random is okay for exploration.
                dir = random.randint(0, 3)
                
                row = np.zeros(num_vars)
                if dir == 0:
                    row[3*i] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j] = -1.0
                    row[3*j+2] = 0.5
                elif dir == 1:
                    row[3*i] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j] = 1.0
                    row[3*j+2] = -0.5
                elif dir == 2:
                    row[3*i+1] = 1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = -1.0
                    row[3*j+2] = 0.5
                else: # dir == 3
                    row[3*i+1] = -1.0
                    row[3*i+2] = 0.5
                    row[3*j+1] = 1.0
                    row[3*j+2] = -0.5
                    
                A_list.append(row)
                b_list.append(0.0)
                
        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        
        # Bounds: s_i > 0, x,y free (but constrained by A_ub)
        # Actually x,y are bounded by the square constraints, so we can set them to (-inf, inf) 
        # or tighter bounds like (0, 1) to help solver.
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 1)) # s (must be positive)
            
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
                
                # Verify validity (LP might have numerical errors)
                if is_valid(centers, sides):
                    score = np.sum(sides)
                    return centers, sides, score
        except Exception:
            pass
            
        return None, None, -np.inf

    # Strategy:
    # 1. Try a few "structured" topologies (Grids)
    # 2. Try many random topologies
    
    # Structured Topology 1: 3x3 Grid + 1 small
    # This is hard to encode as a single LP topology easily without fixing the "small" one's position.
    # Instead, we rely on random topologies to discover good structures.
    # However, random topologies are very likely to be infeasible or yield small squares.
    
    # Better Approach: Use "Sweep" topologies.
    # Imagine sweeping a vertical line. The order of squares intersected.
    # This is complex.
    
    # Let's stick to Random Topologies but with a "Bias" towards grid-like structures.
    # If we assume a 3x3 grid, squares in the same row are left/right, same col are above/below.
    
    def generate_grid_topology(rows, cols):
        """
        Generate a topology consistent with a rows x cols grid.
        Squares are indexed 0..n-1.
        We assign each square a (r, c) in the grid.
        """
        # This function is a placeholder for a more complex logic.
        # For now, we just do random, but we will add a "Grid Bias" by 
        # pre-defining the relative order of a subset of squares.
        pass

    # Let's try a different approach for the "Structured" part:
    # We know the 3x3 grid is valid. 
    # We can start with the 3x3 grid solution and use the LP to *refine* it?
    # No, LP finds global opt for fixed topology.
    
    # Let's try to explicitly construct a few promising topologies.
    
    # Topology A: 3x3 grid with the 10th square in the center? No, center is occupied.
    # Topology B: 2x5 grid.
    
    # Actually, the most effective strategy for LP is to iterate over random topologies 
    # and keep the best feasible one. The number of topologies is huge, but 
    # many random ones will be infeasible.
    
    # To improve hit rate, we can use a "Soft" topology where we allow some 
    # "touching" or "overlapping" in the constraint generation? No, that breaks LP linearity.
    
    # Alternative: Use the SA result as a seed for the topology?
    # The current program's SA finds a solution. We can extract the topology from it 
    # and solve the LP for that exact topology. This guarantees we find the local 
    # optimum of that topology, which is likely better than the SA approximation.
    
    # So, Step 1: Run a quick SA to get a "good enough" valid configuration.
    # Step 2: Extract the topology (relative positions) from that configuration.
    # Step 3: Solve LP for that topology.
    # Step 4: Also try random topologies to explore other structures.
    
    # --- Quick SA to get a seed topology ---
    def run_quick_sa(seed, time_limit=10.0):
        random.seed(seed)
        np.random.seed(seed)
        
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 3x3 grid init
        side_grid = 1.0 / 3.0
        for i in range(9):
            col = i % 3
            row = i // 3
            centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
            sides[i] = side_grid
        centers[9] = [0.9, 0.9]
        sides[9] = 0.1
        
        centers += np.random.normal(0, 0.01, (n, 2))
        
        def evaluate(centers, sides):
            score = np.sum(sides)
            tol = 1e-7
            penalty = 0.0
            for i in range(n):
                cx, cy = centers[i]
                s = sides[i]
                if s <= 0: penalty += 100.0; continue
                if cx - s/2 < -tol: penalty += 100.0 * (-tol - (cx - s/2))
                if cx + s/2 > 1 + tol: penalty += 100.0 * ((cx + s/2) - (1 + tol))
                if cy - s/2 < -tol: penalty += 100.0 * (-tol - (cy - s/2))
                if cy + s/2 > 1 + tol: penalty += 100.0 * ((cy + s/2) - (1 + tol))
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2 - tol
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        penalty += 100.0 * (t - dx) * (t - dy)
            return score - penalty

        current_score = evaluate(centers, sides)
        T = 0.05
        alpha = 0.995
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            for _ in range(50):
                if time.time() > end_time: break
                i = random.randint(0, n-1)
                dx = random.gauss(0, 0.02)
                dy = random.gauss(0, 0.02)
                ds = random.gauss(0, 0.01)
                new_centers = centers.copy()
                new_sides = sides.copy()
                new_centers[i, 0] += dx
                new_centers[i, 1] += dy
                new_sides[i] = max(1e-5, new_sides[i] + ds)
                new_score = evaluate(new_centers, new_sides)
                if new_score > current_score:
                    centers = new_centers
                    sides = new_sides
                    current_score = new_score
                else:
                    delta = current_score - new_score
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        centers = new_centers
                        sides = new_sides
                        current_score = new_score
            T *= alpha
            if T < 1e-6: break
            
        if is_valid(centers, sides):
            return centers, sides
        return None, None

    # Run a few quick SAs to get diverse topologies
    topologies_to_solve = []
    
    for sa_seed in range(10):
        if time.time() > deadline - 100: break
        c, s = run_quick_sa(sa_seed, 5.0)
        if c is not None:
            # Extract topology
            # For each pair (i,j), determine which constraint is active (or closest to active)
            # We just record the "dominant" direction based on the current positions.
            # If x_i + s_i/2 <= x_j - s_j/2 + eps, then 'left'
            # If x_j + s_j/2 <= x_i - s_i/2 + eps, then 'right'
            # If y_i + s_i/2 <= y_j - s_j/2 + eps, then 'below'
            # If y_j + s_j/2 <= y_i - s_i/2 + eps, then 'above'
            # If multiple, pick one arbitrarily or the one with largest margin?
            # For LP, we need to pick ONE valid constraint that is satisfied.
            # Since the config is valid, at least one is satisfied.
            
            topo = []
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    # Check which constraints are satisfied
                    sat = []
                    # Left
                    if c[i,0] + s[i]/2 <= c[j,0] - s[j]/2 + 1e-9: sat.append(0)
                    # Right
                    if c[j,0] + s[j]/2 <= c[i,0] - s[i]/2 + 1e-9: sat.append(1)
                    # Below
                    if c[i,1] + s[i]/2 <= c[j,1] - s[j]/2 + 1e-9: sat.append(2)
                    # Above
                    if c[j,1] + s[j]/2 <= c[i,1] - s[i]/2 + 1e-9: sat.append(3)
                    
                    if not sat:
                        valid = False
                        break
                    # Pick the one with the largest "slack" (safety margin) to ensure LP feasibility
                    # Slack for Left: (c[j,0] - s[j]/2) - (c[i,0] + s[i]/2)
                    slacks = []
                    if 0 in sat: slacks.append((c[j,0] - s[j]/2) - (c[i,0] + s[i]/2))
                    if 1 in sat: slacks.append((c[i,0] - s[i]/2) - (c[j,0] + s[j]/2))
                    if 2 in sat: slacks.append((c[j,1] - s[j]/2) - (c[i,1] + s[i]/2))
                    if 3 in sat: slacks.append((c[i,1] - s[i]/2) - (c[j,1] + s[j]/2))
                    
                    # Map slack to direction
                    # We need to know which direction corresponds to which slack index
                    # Let's just pick the first one for simplicity, or the max slack.
                    # To be robust, let's just pick the first satisfied one.
                    topo.append(sat[0])
                if not valid: break
            
            if valid:
                topologies_to_solve.append(topo)
    
    # Solve LP for each extracted topology
    for topo in topologies_to_solve:
        if time.time() > deadline - 20: break
        # Re-implement solve_topology to take a specific topology
        # We can reuse the logic by passing the fixed directions
        # Instead of random.randint, we use the pre-computed dir
        
        # We'll write a helper that takes a list of directions
        def solve_specific_topology(dirs):
            num_vars = 3 * n
            c = np.zeros(num_vars)
            for i in range(n):
                c[3*i + 2] = -1.0
                
            A_list = []
            b_list = []
            
            for i in range(n):
                row = np.zeros(num_vars); row[3*i] = -1.0; row[3*i+2] = 0.5; A_list.append(row); b_list.append(0.0)
                row = np.zeros(num_vars); row[3*i] = 1.0; row[3*i+2] = 0.5; A_list.append(row); b_list.append(1.0)
                row = np.zeros(num_vars); row[3*i+1] = -1.0; row[3*i+2] = 0.5; A_list.append(row); b_list.append(0.0)
                row = np.zeros(num_vars); row[3*i+1] = 1.0; row[3*i+2] = 0.5; A_list.append(row); b_list.append(1.0)
                
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    dir = dirs[idx]
                    idx += 1
                    row = np.zeros(num_vars)
                    if dir == 0:
                        row[3*i] = 1.0; row[3*i+2] = 0.5; row[3*j] = -1.0; row[3*j+2] = 0.5
                    elif dir == 1:
                        row[3*i] = -1.0; row[3*i+2] = 0.5; row[3*j] = 1.0; row[3*j+2] = -0.5
                    elif dir == 2:
                        row[3*i+1] = 1.0; row[3*i+2] = 0.5; row[3*j+1] = -1.0; row[3*j+2] = 0.5
                    else:
                        row[3*i+1] = -1.0; row[3*i+2] = 0.5; row[3*j+1] = 1.0; row[3*j+2] = -0.5
                    A_list.append(row)
                    b_list.append(0.0)
                    
            A_ub = np.array(A_list)
            b_ub = np.array(b_list)
            bounds = [(0, 1), (0, 1), (1e-6, 1)] * n
            
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
                        return centers, sides, np.sum(sides)
            except:
                pass
            return None, None, -np.inf

        c, s, sc = solve_specific_topology(topo)
        if c is not None and sc > best_score:
            best_centers = c
            best_sides = s
            best_score = sc

    # Also try a few purely random topologies to explore new structures
    for _ in range(50):
        if time.time() > deadline - 20: break
        random_dirs = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        c, s, sc = solve_specific_topology(random_dirs)
        if c is not None and sc > best_score:
            best_centers = c
            best_sides = s
            best_score = sc

    # Fallback to 3x3 grid if no LP solution found
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

    return best_centers, best_sides

# EVOLVE-BLOCK-END
