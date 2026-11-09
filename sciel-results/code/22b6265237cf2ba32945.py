# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from itertools import product

_TOL = 1e-7
_DEADLINE_S = 620.0


def _feasible(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _grid_init(n=10):
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _solve_lp_for_pattern(n, pattern, start_time, deadline):
    """
    Solve LP given a fixed separation pattern.
    pattern: list of length num_pairs, each entry is 0,1,2,3 for x_neg, x_pos, y_neg, y_pos
    Returns (centers, sides) or None.
    """
    if time.time() > deadline:
        return None
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    # Variables: cx[0..n-1], cy[0..n-1], s[0..n-1]
    # Total: 3n variables
    cx_idx = np.arange(n)
    cy_idx = np.arange(n, 2 * n)
    s_idx = np.arange(2 * n, 3 * n)
    total_vars = 3 * n
    
    # Objective: maximize sum(s) -> minimize -sum(s)
    c = np.zeros(total_vars)
    for i in range(n):
        c[s_idx[i]] = -1.0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Boundary constraints
    for i in range(n):
        # cx_i - 0.5 s_i >= 0
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[s_idx[i]] = -0.5
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # cx_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[s_idx[i]] = 0.5
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)
        
        # cy_i - 0.5 s_i >= 0
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[s_idx[i]] = -0.5
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # cy_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[s_idx[i]] = 0.5
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)
    
    # Separation constraints based on pattern
    for k, ((i, j), sep) in enumerate(zip(pairs, pattern)):
        if sep == 0:  # x_neg: cx_j - cx_i >= 0.5(si+sj)
            row = np.zeros(total_vars)
            row[cx_idx[j]] = 1.0
            row[cx_idx[i]] = -1.0
            row[s_idx[i]] = -0.5
            row[s_idx[j]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
        elif sep == 1:  # x_pos: cx_i - cx_j >= 0.5(si+sj)
            row = np.zeros(total_vars)
            row[cx_idx[i]] = 1.0
            row[cx_idx[j]] = -1.0
            row[s_idx[i]] = -0.5
            row[s_idx[j]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
        elif sep == 2:  # y_neg: cy_j - cy_i >= 0.5(si+sj)
            row = np.zeros(total_vars)
            row[cy_idx[j]] = 1.0
            row[cy_idx[i]] = -1.0
            row[s_idx[i]] = -0.5
            row[s_idx[j]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
        elif sep == 3:  # y_pos: cy_i - cy_j >= 0.5(si+sj)
            row = np.zeros(total_vars)
            row[cy_idx[i]] = 1.0
            row[cy_idx[j]] = -1.0
            row[s_idx[i]] = -0.5
            row[s_idx[j]] = -0.5
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(np.inf)
    
    A = np.array(A_rows)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    lb[s_idx] = 1e-6
    ub[s_idx] = 1.0
    
    try:
        res = linprog(c, A_ub=A, b_ub=b_upper, A_eq=None, b_eq=None,
                      bounds=list(zip(lb, ub)), method='highs')
        if res.success:
            cx = res.x[cx_idx]
            cy = res.x[cy_idx]
            s = res.x[s_idx]
            centers = np.stack([cx, cy], axis=1)
            return centers, s
    except Exception:
        pass
    return None


def _try_structured_patterns(n, start_time, deadline):
    """
    Try structured separation patterns and return the best feasible solution.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    
    best_score = -np.inf
    best_result = None
    
    # Pattern 1: All horizontal separation (x direction)
    # Sort by x-coordinate and separate consecutively
    for sep_dir in range(4):  # Try all 4 base directions
        pattern = [sep_dir] * num_pairs
        result = _solve_lp_for_pattern(n, pattern, start_time, deadline)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_result = (centers, sides)
    
    # Pattern 2: Grid-like - separate in x for same row, y for same column
    # 2x5 grid
    if time.time() < deadline:
        # Create a 2x5 grid pattern
        # Squares 0-4 in row 0, 5-9 in row 1
        # In each row, separate in x; between rows, separate in y
        pattern = []
        for i in range(n):
            for j in range(i + 1, n):
                row_i = i // 5
                row_j = j // 5
                if row_i == row_j:
                    # Same row, separate in x
                    if i < j:
                        pattern.append(1)  # x_pos: cx_i - cx_j... wait, need i left of j
                        # Actually for i<j in same row, i should be left: cx_i < cx_j
                        # So cx_j - cx_i >= ... => x_neg (sep=0)
                        pattern[-1] = 0
                else:
                    # Different rows, separate in y
                    if row_i < row_j:
                        # i above j: cy_i < cy_j => cy_j - cy_i >= ... => y_neg (sep=2)
                        pattern.append(2)
                    else:
                        pattern.append(3)
        
        result = _solve_lp_for_pattern(n, pattern, start_time, deadline)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_result = (centers, sides)
    
    # Pattern 3: 5x2 grid
    if time.time() < deadline:
        pattern = []
        for i in range(n):
            for j in range(i + 1, n):
                col_i = i % 2
                col_j = j % 2
                if col_i == col_j:
                    # Same column, separate in y
                    if i < j:
                        # i above j: cy_j - cy_i >= ... => y_neg (sep=2)
                        pattern.append(2)
                else:
                    # Different columns, separate in x
                    if col_i < col_j:
                        # i left of j: cx_j - cx_i >= ... => x_neg (sep=0)
                        pattern.append(0)
                    else:
                        pattern.append(1)
        
        result = _solve_lp_for_pattern(n, pattern, start_time, deadline)
        if result is not None:
            centers, sides = result
            if _feasible(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_result = (centers, sides)
    
    # Pattern 4: Random patterns with some structure
    if time.time() < deadline:
        np.random.seed(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            # Generate a random pattern but with some consistency
            pattern = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Choose based on relative position heuristic
                    if np.random.random() < 0.5:
                        pattern.append(0)  # x_neg
                    elif np.random.random() < 0.5:
                        pattern.append(1)  # x_pos
                    elif np.random.random() < 0.5:
                        pattern.append(2)  # y_neg
                    else:
                        pattern.append(3)  # y_pos
            
            result = _solve_lp_for_pattern(n, pattern, start_time, deadline)
            if result is not None:
                centers, sides = result
                if _feasible(centers, sides):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_result = (centers, sides)
    
    return best_result


def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + _DEADLINE_S - 10
    
    # Try structured patterns
    result = _try_structured_patterns(n, start_time, deadline)
    if result is not None:
        centers, sides = result
        if _feasible(centers, sides):
            return centers, sides
    
    # Fallback to grid
    centers, sides = _grid_init(n)
    if _feasible(centers, sides):
        return centers, sides
    
    # Last resort
    centers = np.zeros((n, 2))
    sides = np.full(n, 0.1)
    return centers, sides


# EVOLVE-BLOCK-END
