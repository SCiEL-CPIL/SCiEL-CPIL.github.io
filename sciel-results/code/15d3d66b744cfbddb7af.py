# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import combinations
import random

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    
    Strategy:
    1. Represent the packing configuration by a set of separation constraints 
       (directed graph edges: i <_x j or i <_y j).
    2. For a given set of constraints, solve a Linear Program (LP) to maximize 
       the sum of side lengths. The LP variables are side lengths s_i and 
       positions cx_i, cy_i.
    3. Use a local search (Hill Climbing) on the constraint graph. 
       - Start with a valid "grid" graph.
       - In each step, try to flip a random constraint or add/remove constraints 
         to improve the LP solution.
    4. This avoids the local minima issues of SLSQP by leveraging the exact 
       linear structure of the problem for each discrete topology.
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)
    
    # Map pair index to (i, j)
    pair_map = {k: pairs[k] for k in range(n_pairs)}

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def solve_lp(constraints_mask):
        """
        constraints_mask: boolean array of length n_pairs.
        True means we enforce separation. But which direction?
        We need to encode direction. 
        Let's use two arrays: sep_x and sep_y.
        Actually, a better representation for LP:
        For each pair (i,j), we must have EITHER x_i + s_i/2 <= x_j - s_j/2 
        OR y_i + s_i/2 <= y_j - s_j/2.
        
        This is a disjunctive constraint. We cannot directly put this in a single LP.
        We must fix the "orientation" for each pair.
        
        Revised Representation:
        orientation: array of length n_pairs.
        0: i is left of j (x_i + s_i/2 <= x_j - s_j/2)
        1: i is below j (y_i + s_i/2 <= y_j - s_j/2)
        
        Wait, the current program's "pattern" is exactly this. 
        The issue with the current program is that it uses SLSQP.
        Let's switch to LP.
        
        Variables:
        x = [s_0..s_9, cx_0..cx_9, cy_0..cy_9]
        Size: 30 variables.
        
        Objective: Maximize sum(s_i) -> Minimize -sum(s_i)
        
        Constraints:
        1. Boundary:
           cx_i - s_i/2 >= 0
           cx_i + s_i/2 <= 1
           cy_i - s_i/2 >= 0
           cy_i + s_i/2 <= 1
           
        2. Separation (based on orientation):
           If orientation[k] == 0 (x-sep, i left of j):
               cx_j - cx_i - (s_i + s_j)/2 >= 0
           If orientation[k] == 1 (y-sep, i below j):
               cy_j - cy_i - (s_i + s_j)/2 >= 0
               
        3. Positivity:
           s_i > 0 (handled by bounds)
        
        We can use scipy.optimize.linprog.
        """
        if time.time() > deadline - 5:
            return None, None

        # Objective: c^T x = -sum(s)
        c = np.zeros(3 * n)
        c[:n] = -1.0  # Minimize -sum(s)

        # Bounds: s_i >= 0, cx_i in [0,1], cy_i in [0,1]
        # Actually, cx can be in [0,1] if s is small enough, but let's use [0,1] for simplicity
        # Wait, cx - s/2 >= 0 and cx + s/2 <= 1 implies cx in [s/2, 1-s/2].
        # It is safer to let cx, cy be in [0,1] and enforce boundary constraints.
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # s_i
        for i in range(n):
            bounds.append((0, 1))  # cx_i
        for i in range(n):
            bounds.append((0, 1))  # cy_i

        # Inequality Constraints: A_ub x <= b_ub
        # We have "greater than or equal to" constraints.
        # Convert to <= by multiplying by -1.
        
        A_rows = []
        b_rows = []
        
        # Boundary constraints:
        # cx_i - s_i/2 >= 0  =>  s_i/2 - cx_i <= 0
        # cx_i + s_i/2 <= 1  =>  cx_i + s_i/2 <= 1
        # cy_i - s_i/2 >= 0  =>  s_i/2 - cy_i <= 0
        # cy_i + s_i/2 <= 1  =>  cy_i + s_i/2 <= 1
        
        for i in range(n):
            # s_i/2 - cx_i <= 0
            row = np.zeros(3 * n)
            row[i] = 0.5
            row[n + i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cx_i + s_i/2 <= 1
            row = np.zeros(3 * n)
            row[i] = 0.5
            row[n + i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)
            
            # s_i/2 - cy_i <= 0
            row = np.zeros(3 * n)
            row[i] = 0.5
            row[2 * n + i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
            
            # cy_i + s_i/2 <= 1
            row = np.zeros(3 * n)
            row[i] = 0.5
            row[2 * n + i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)

        # Separation constraints
        for k in range(n_pairs):
            i, j = pair_map[k]
            if orientation is not None and orientation[k] == 0:
                # i left of j: cx_j - cx_i - (s_i + s_j)/2 >= 0
                # => (s_i + s_j)/2 - cx_j + cx_i <= 0
                row = np.zeros(3 * n)
                row[i] = 0.5
                row[j] = 0.5
                row[n + i] = 1.0
                row[n + j] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            else:
                # i below j: cy_j - cy_i - (s_i + s_j)/2 >= 0
                # => (s_i + s_j)/2 - cy_j + cy_i <= 0
                row = np.zeros(3 * n)
                row[i] = 0.5
                row[j] = 0.5
                row[2 * n + i] = 1.0
                row[2 * n + j] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)

        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                val = -res.fun
                s = res.x[:n]
                cx = res.x[n:2*n]
                cy = res.x[2*n:3*n]
                centers = np.column_stack([cx, cy])
                # Check if strictly valid (LP might sit on boundary, which is allowed)
                if is_valid(centers, s):
                    return val, res.x
        except Exception:
            pass
        return None, None

    # --- Initial Orientation Generation ---
    # We need a valid initial orientation.
    # A simple one is a grid.
    # Let's create a 3x3 grid + 1.
    # Or just random orientations and see if LP is feasible.
    # If LP is infeasible, the orientation contains a cycle or contradiction.
    
    best_val = 0.0
    best_x = None
    best_orientation = None
    
    # Try a few random orientations to find a feasible start
    for _ in range(50):
        if time.time() > deadline - 100:
            break
        # Random orientation
        orient = np.random.randint(0, 2, size=n_pairs)
        val, x = solve_lp(orient)
        if val is not None and val > best_val:
            best_val = val
            best_x = x
            best_orientation = orient.copy()
            
    # If no feasible orientation found, fallback to a known good structure (e.g., all x-sep is infeasible for n>1)
    # Let's try a "snake" pattern or grid pattern explicitly.
    if best_orientation is None:
        # Try a 3x3 grid pattern
        orient = np.zeros(n_pairs, dtype=int)
        for k, (i, j) in enumerate(pairs):
            row_i = i // 3
            row_j = j // 3
            if row_i != row_j:
                orient[k] = 1 # y-sep
            else:
                orient[k] = 0 # x-sep
        val, x = solve_lp(orient)
        if val is not None:
            best_val = val
            best_x = x
            best_orientation = orient.copy()
            
    if best_orientation is None:
        # Absolute fallback: 4x4 grid squares (side 0.25)
        # This is always valid.
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    # --- Local Search on Orientation ---
    # We will try to flip edges in the orientation to improve the LP solution.
    # If flipping makes the LP infeasible, we discard the flip.
    
    current_orient = best_orientation.copy()
    current_val = best_val
    current_x = best_x
    
    max_iters = 5000
    iter_count = 0
    
    while iter_count < max_iters and time.time() < deadline - 20:
        iter_count += 1
        
        # Pick a random pair to flip
        k = np.random.randint(n_pairs)
        new_orient = current_orient.copy()
        new_orient[k] = 1 - new_orient[k]
        
        # Solve LP with new orientation
        # Use previous solution as hint? Linprog doesn't support warm start directly in scipy wrapper easily,
        # but HiGHS is fast enough for 30 variables.
        new_val, new_x = solve_lp(new_orient)
        
        if new_val is not None:
            if new_val > current_val:
                # Accept improvement
                current_orient = new_orient
                current_val = new_val
                current_x = new_x
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = current_x
                    best_orientation = current_orient.copy()
            else:
                # Simulated Annealing style acceptance? 
                # Since LP is deterministic for a given graph, and we want to escape local minima,
                # we might accept worse solutions with low probability.
                # However, infeasibility is the bigger risk.
                # Let's stick to strict hill climbing first, or add a small probability of acceptance.
                if np.random.random() < 0.05:
                    current_orient = new_orient
                    current_val = new_val
                    current_x = new_x
        else:
            # Infeasible, discard
            pass

    # Final refinement: try all single flips from the best orientation found
    # to ensure we are at a local optimum.
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for k in range(n_pairs):
                if time.time() > deadline - 5:
                    break
                new_orient = best_orientation.copy()
                new_orient[k] = 1 - new_orient[k]
                new_val, new_x = solve_lp(new_orient)
                if new_val is not None and new_val > best_val:
                    best_val = new_val
                    best_x = new_x
                    best_orientation = new_orient
                    improved = True

    # Extract final result
    if best_x is not None:
        s = best_x[:n]
        cx = best_x[n:2*n]
        cy = best_x[2*n:3*n]
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = s.astype(np.float64)
        
        # Final validity check
        if not is_valid(centers, sides):
            # Fallback to grid
            k = 4
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            centers = np.stack([cx, cy], axis=1).astype(np.float64)
            sides = np.full(n, side, dtype=np.float64)
    else:
        # Fallback to grid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
