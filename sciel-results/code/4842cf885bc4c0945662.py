# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time
import itertools
from collections import deque


def _is_valid(centers, sides, tol=1e-7):
    """Check if the placement is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check boundaries
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i, s_j = sides[i], sides[j]
            t = (s_i + s_j) / 2 - tol
            if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                return False
    return True


def _solve_lp_for_separations(sep_x, sep_y, n=10, tol=1e-7):
    """
    Given separation patterns:
      sep_x[i][j] = 1 means square i is to the LEFT of square j (i's right edge <= j's left edge)
      sep_y[i][j] = 1 means square i is BELOW square j (i's top edge <= j's bottom edge)
    
    For each pair (i,j), exactly one of sep_x[i][j], sep_x[j][i], sep_y[i][j], sep_y[j][i] must be 1.
    
    We solve an LP: maximize sum(s_i) subject to:
      - Boundary constraints: s_i/2 <= cx_i <= 1 - s_i/2, similarly for y
      - Separation constraints based on the pattern
    
    Variables: cx_i, cy_i, s_i for i in 0..n-1
    Total: 3n variables.
    
    We maximize sum(s_i), which is equivalent to minimizing -sum(s_i).
    """
    # Variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_{n-1}, cy_{n-1}, s_{n-1}]
    def var(i, dim):
        return 3*i + dim  # dim: 0=cx, 1=cy, 2=s
    
    # Objective: minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[var(i, 2)] = -1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_rows = []
    
    # Boundary constraints:
    # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
    # cx_i + s_i/2 <= 1  =>   cx_i - s_i/2 <= 1
    # Similarly for y
    for i in range(n):
        # -cx_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[var(i, 0)] = -1.0
        row[var(i, 2)] = 0.5
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
        # cx_i - s_i/2 <= 1
        row = np.zeros(3*n)
        row[var(i, 0)] = 1.0
        row[var(i, 2)] = -0.5
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)
        
        # -cy_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[var(i, 1)] = -1.0
        row[var(i, 2)] = 0.5
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
        # cy_i - s_i/2 <= 1
        row = np.zeros(3*n)
        row[var(i, 1)] = 1.0
        row[var(i, 2)] = -0.5
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)
        
        # s_i >= 0 is handled by bounds; also s_i > 0 enforced by bounds
    
    # Separation constraints:
    # If sep_x[i][j] == 1: square i left of square j
    #   cx_i + s_i/2 <= cx_j - s_j/2
    #   => cx_i - cx_j + s_i/2 + s_j/2 <= 0
    for i in range(n):
        for j in range(i+1, n):
            if sep_x[i][j] == 1:
                row = np.zeros(3*n)
                row[var(i, 0)] = 1.0
                row[var(j, 0)] = -1.0
                row[var(i, 2)] = 0.5
                row[var(j, 2)] = 0.5
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
            elif sep_x[j][i] == 1:
                row = np.zeros(3*n)
                row[var(j, 0)] = 1.0
                row[var(i, 0)] = -1.0
                row[var(j, 2)] = 0.5
                row[var(i, 2)] = 0.5
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
            elif sep_y[i][j] == 1:
                row = np.zeros(3*n)
                row[var(i, 1)] = 1.0
                row[var(j, 1)] = -1.0
                row[var(i, 2)] = 0.5
                row[var(j, 2)] = 0.5
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
            elif sep_y[j][i] == 1:
                row = np.zeros(3*n)
                row[var(j, 1)] = 1.0
                row[var(i, 1)] = -1.0
                row[var(j, 2)] = 0.5
                row[var(i, 2)] = 0.5
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
            else:
                # No separation specified - this shouldn't happen
                return None, None
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Bounds: s_i >= 0, cx_i, cy_i can be anything (bounded by constraints)
    bounds = []
    for i in range(n):
        bounds.append((None, None))  # cx_i
        bounds.append((None, None))  # cy_i
        bounds.append((1e-10, None))  # s_i > 0
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None, None
    
    if not result.success:
        return None, None
    
    x = result.x
    centers = np.array([[x[var(i,0)], x[var(i,1)]] for i in range(n)])
    sides = np.array([x[var(i,2)] for i in range(n)])
    
    return centers, sides


def _generate_separation_patterns(n=10, max_patterns=200):
    """
    Generate diverse separation patterns using a greedy/randomized approach.
    For each pair (i,j), choose one of 4 directions.
    We use several strategies to generate diverse patterns.
    """
    patterns = []
    
    # Strategy 1: Grid-based patterns
    # 3x4 grid or 4x3 grid
    for k_rows in [2, 3, 4]:
        for k_cols in [3, 4, 5]:
            if k_rows * k_cols < n:
                continue
            sep_x = np.zeros((n, n), dtype=int)
            sep_y = np.zeros((n, n), dtype=int)
            idx = np.arange(n)
            cols = idx % k_cols
            rows = idx // k_cols
            for i in range(n):
                for j in range(i+1, n):
                    if cols[i] < cols[j]:
                        sep_x[i][j] = 1
                    elif cols[i] > cols[j]:
                        sep_x[j][i] = 1
                    else:
                        if rows[i] < rows[j]:
                            sep_y[i][j] = 1
                        else:
                            sep_y[j][i] = 1
            patterns.append((sep_x, sep_y))
    
    # Strategy 2: Random patterns with some structure
    np.random.seed(42)
    for trial in range(max_patterns - len(patterns)):
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        # Assign random "columns" and "rows" to create a partial order
        col_order = np.random.permutation(n)
        row_order = np.random.permutation(n)
        col_rank = np.zeros(n, dtype=int)
        row_rank = np.zeros(n, dtype=int)
        for rank, sq in enumerate(col_order):
            col_rank[sq] = rank
        for rank, sq in enumerate(row_order):
            row_rank[sq] = rank
        
        for i in range(n):
            for j in range(i+1, n):
                # Decide based on relative ranks with some randomness
                if col_rank[i] < col_rank[j] and np.random.random() < 0.7:
                    sep_x[i][j] = 1
                elif col_rank[j] < col_rank[i] and np.random.random() < 0.7:
                    sep_x[j][i] = 1
                elif row_rank[i] < row_rank[j] and np.random.random() < 0.7:
                    sep_y[i][j] = 1
                else:
                    # Fallback
                    if row_rank[j] < row_rank[i]:
                        sep_y[j][i] = 1
                    else:
                        sep_y[i][j] = 1
        
        # Verify each pair has exactly one constraint
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                count = sep_x[i][j] + sep_x[j][i] + sep_y[i][j] + sep_y[j][i]
                if count != 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            patterns.append((sep_x, sep_y))
    
    # Strategy 3: Spiral / snake patterns
    # Snake pattern in a grid
    for k in [3, 4]:
        sep_x = np.zeros((n, n), dtype=int)
        sep_y = np.zeros((n, n), dtype=int)
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        for i in range(n):
            for j in range(i+1, n):
                # Snake: alternate direction per row
                if rows[i] != rows[j]:
                    if rows[i] < rows[j]:
                        sep_y[i][j] = 1
                    else:
                        sep_y[j][i] = 1
                else:
                    # Same row, snake direction
                    if rows[i] % 2 == 0:
                        if cols[i] < cols[j]:
                            sep_x[i][j] = 1
                        else:
                            sep_x[j][i] = 1
                    else:
                        if cols[i] > cols[j]:
                            sep_x[i][j] = 1
                        else:
                            sep_x[j][i] = 1
        patterns.append((sep_x, sep_y))
    
    return patterns[:max_patterns]


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses combinatorial separation patterns + LP optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Baseline: 3x3 grid
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_value = float(np.sum(best_sides))
    
    # Generate and try separation patterns
    patterns = _generate_separation_patterns(n, max_patterns=300)
    
    for pi, (sep_x, sep_y) in enumerate(patterns):
        if time.time() > deadline:
            break
        
        centers, sides = _solve_lp_for_separations(sep_x, sep_y, n)
        
        if centers is None:
            continue
        
        if not _is_valid(centers, sides):
            continue
        
        value = float(np.sum(sides))
        if value > best_value:
            best_value = value
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # Local refinement: try small perturbations of the best solution
    if time.time() < deadline - 60:
        # Try swapping some separation constraints around the best solution
        np.random.seed(123)
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Start from best solution and try to improve by relaxing one constraint
            # and re-solving with a modified pattern
            # For simplicity, try random small modifications
            
            # Create a perturbed pattern based on the best solution's implied order
            sep_x = np.zeros((n, n), dtype=int)
            sep_y = np.zeros((n, n), dtype=int)
            
            for i in range(n):
                for j in range(i+1, n):
                    # Determine separation from current best solution
                    dx = best_centers[j][0] - best_centers[i][0]
                    dy = best_centers[j][1] - best_centers[i][1]
                    gap_x = dx - (best_sides[i] + best_sides[j])/2
                    gap_y = dy - (best_sides[i] + best_sides[j])/2
                    
                    # With some probability, flip the constraint
                    if np.random.random() < 0.1:
                        # Random choice
                        choice = np.random.randint(4)
                    else:
                        # Keep the implied direction
                        if gap_x >= 0 and gap_x >= gap_y:
                            choice = 0  # i left of j
                        elif gap_x <= 0 and -gap_x >= gap_y:
                            choice = 1  # j left of i
                        elif gap_y >= 0 and gap_y >= gap_x:
                            choice = 2  # i below j
                        else:
                            choice = 3  # j below i
                    
                    if choice == 0:
                        sep_x[i][j] = 1
                    elif choice == 1:
                        sep_x[j][i] = 1
                    elif choice == 2:
                        sep_y[i][j] = 1
                    else:
                        sep_y[j][i] = 1
            
            centers, sides = _solve_lp_for_separations(sep_x, sep_y, n)
            if centers is None:
                continue
            if not _is_valid(centers, sides):
                continue
            value = float(np.sum(sides))
            if value > best_value:
                best_value = value
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
