# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import permutations, combinations

def squares_in_square10():
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    
    Strategy:
    1. Enumerate many structured separation patterns (which axis separates each pair)
    2. For each pattern, solve a Linear Program (LP) to find optimal side lengths and positions
    3. Return the best feasible solution found
    """
    t0 = time.time()
    deadline = t0 + 590.0  # safety margin before 630s kill
    
    n = 10
    tol = 1e-7
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs)  # 45
    
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
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,0]) < t:
                    return False
        return True
    
    def solve_lp(separation_pattern):
        """
        Solve LP for given separation pattern.
        Variables: s_0..s_9 (sides), cx_0..cx_9, cy_0..cy_9
        Total: 30 variables
        Objective: maximize sum(s_i)
        
        separation_pattern: array of length n_pairs, 0=x-sep, 1=y-sep
        """
        # Variable indices:
        # s: 0-9, cx: 10-19, cy: 20-29
        s_idx = np.arange(0, n)
        cx_idx = np.arange(n, 2*n)
        cy_idx = np.arange(2*n, 3*n)
        
        # Objective: maximize sum(s_i) -> minimize -sum(s_i)
        c = np.zeros(3*n)
        c[s_idx] = -1.0
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_vals = []
        
        # Boundary constraints:
        # cx_i - s_i/2 >= 0  =>  -cx_i + s_i/2 <= 0
        # cx_i + s_i/2 <= 1  =>  cx_i + s_i/2 <= 1
        # cy_i - s_i/2 >= 0  =>  -cy_i + s_i/2 <= 0
        # cy_i + s_i/2 <= 1  =>  cy_i + s_i/2 <= 1
        for i in range(n):
            # -cx_i + s_i/2 <= 0
            row = np.zeros(3*n)
            row[cx_idx[i]] = -1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_vals.append(0.0)
            
            # cx_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[cx_idx[i]] = 1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_vals.append(1.0)
            
            # -cy_i + s_i/2 <= 0
            row = np.zeros(3*n)
            row[cy_idx[i]] = -1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_vals.append(0.0)
            
            # cy_i + s_i/2 <= 1
            row = np.zeros(3*n)
            row[cy_idx[i]] = 1.0
            row[s_idx[i]] = 0.5
            A_rows.append(row)
            b_vals.append(1.0)
        
        # Separation constraints:
        for k, (i, j) in enumerate(pairs):
            if separation_pattern[k] == 0:
                # i is left of j: cx_j - cx_i >= (s_i + s_j)/2
                # cx_j - cx_i - (s_i + s_j)/2 >= 0
                # -(cx_j - cx_i - (s_i + s_j)/2) <= 0
                # -cx_j + cx_i + (s_i + s_j)/2 <= 0
                row = np.zeros(3*n)
                row[cx_idx[j]] = -1.0
                row[cx_idx[i]] = 1.0
                row[s_idx[i]] = 0.5
                row[s_idx[j]] = 0.5
                A_rows.append(row)
                b_vals.append(0.0)
            else:
                # i is below j: cy_j - cy_i >= (s_i + s_j)/2
                # cy_j - cy_i - (s_i + s_j)/2 >= 0
                # -cy_j + cy_i + (s_i + s_j)/2 <= 0
                row = np.zeros(3*n)
                row[cy_idx[j]] = -1.0
                row[cy_idx[i]] = 1.0
                row[s_idx[i]] = 0.5
                row[s_idx[j]] = 0.5
                A_rows.append(row)
                b_vals.append(0.0)
        
        A_ub = np.array(A_rows) if A_rows else None
        b_ub = np.array(b_vals) if b_vals else None
        
        # Bounds: s_i >= 0 (and <= 1), cx_i, cy_i in [0, 1]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 1)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                val = -res.fun
                s = res.x[:n]
                cx = res.x[n:2*n]
                cy = res.x[2*n:3*n]
                centers = np.column_stack([cx, cy])
                if is_valid(centers, s):
                    return val, centers, s
        except Exception:
            pass
        return None, None, None
    
    # Generate structured patterns
    patterns = []
    
    # Pattern 1: 3 rows x 3 cols + 1 extra
    # Rows: 0-2, 3-5, 6-8, 9 is extra
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        row_i = min(i, 8) // 3 if i < 9 else 3
        row_j = min(j, 8) // 3 if j < 9 else 3
        if row_i != row_j:
            p[k] = 1  # y-sep
        else:
            p[k] = 0  # x-sep
    patterns.append(p)
    
    # Pattern 2: 4 rows x 2-3 cols
    # Rows: 0-1, 2-3, 4-5, 6-7, 8-9
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        row_i = i // 2
        row_j = j // 2
        if row_i != row_j:
            p[k] = 1
        else:
            p[k] = 0
    patterns.append(p)
    
    # Pattern 3: 2 rows x 5 cols
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        row_i = i // 5
        row_j = j // 5
        if row_i != row_j:
            p[k] = 1
        else:
            p[k] = 0
    patterns.append(p)
    
    # Pattern 4: 5 rows x 2 cols
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        row_i = i // 2
        row_j = j // 2
        if row_i != row_j:
            p[k] = 1
        else:
            p[k] = 0
    patterns.append(p)
    
    # Pattern 5: All x-sep
    p = np.zeros(n_pairs, dtype=int)
    patterns.append(p)
    
    # Pattern 6: All y-sep
    p = np.ones(n_pairs, dtype=int)
    patterns.append(p)
    
    # Pattern 7: Checkerboard-like
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        if (i + j) % 2 == 0:
            p[k] = 0
        else:
            p[k] = 1
    patterns.append(p)
    
    # Pattern 8: Diagonal
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        if i < j:
            p[k] = 0 if (j - i) <= 2 else 1
    patterns.append(p)
    
    # Add random patterns
    np.random.seed(42)
    for _ in range(20):
        p = np.random.randint(0, 2, size=n_pairs)
        patterns.append(p)
    
    # Solve all patterns
    best_val = 0.0
    best_centers = None
    best_sides = None
    
    for idx, pat in enumerate(patterns):
        if time.time() > deadline - 30:
            break
        val, centers, sides = solve_lp(pat)
        if val is not None and val > best_val:
            best_val = val
            best_centers = centers
            best_sides = sides
    
    # Fallback if all failed
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    else:
        best_centers = best_centers.astype(np.float64)
        best_sides = best_sides.astype(np.float64)
        
        # Final validity check
        if not is_valid(best_centers, best_sides):
            k = 4
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
