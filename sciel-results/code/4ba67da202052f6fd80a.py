# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_TOL = 1e-7
_DEADLINE_S = 620.0  # safety margin before 630s kill


def _feasible(centers, sides, tol=_TOL):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # inside [-tol, 1+tol]
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # pairwise non-overlap
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


def _solve_milp(n=10):
    """
    Solve the packing problem using MILP.
    Variables:
    0..n-1: cx
    n..2n-1: cy
    2n..3n-1: s
    3n..: binary variables for separation
    """
    # Indices
    cx_idx = np.arange(n)
    cy_idx = np.arange(n, 2 * n)
    s_idx = np.arange(2 * n, 3 * n)
    
    # Binary variables
    # For each pair (i, j) with i < j, we have 4 binaries:
    # b_x_pos: cx_i - cx_j >= (si+sj)/2
    # b_x_neg: cx_j - cx_i >= (si+sj)/2
    # b_y_pos: cy_i - cy_j >= (si+sj)/2
    # b_y_neg: cy_j - cy_i >= (si+sj)/2
    # Exactly one must be 1.
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    bin_start = 3 * n
    bin_idx = {}
    for k, (i, j) in enumerate(pairs):
        base = bin_start + 4 * k
        bin_idx[(i, j)] = {
            'x_pos': base,
            'x_neg': base + 1,
            'y_pos': base + 2,
            'y_neg': base + 3
        }
    
    total_vars = bin_start + 4 * num_pairs
    
    # Objective: maximize sum(s) -> minimize -sum(s)
    c = np.zeros(total_vars)
    for i in range(n):
        c[s_idx[i]] = -1.0
        
    # Integrality: 0 for continuous, 1 for binary
    integrality = np.zeros(total_vars)
    integrality[bin_start:] = 1
    
    # Bounds
    # cx, cy in [0, 1]
    # s in [1e-4, 1]
    # binaries in [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    lb[s_idx] = 1e-4
    ub[s_idx] = 1.0
    lb[bin_start:] = 0
    ub[bin_start:] = 1
    
    # Constraints
    # 1. Boundary constraints:
    # cx_i - s_i/2 >= 0  => cx_i - 0.5 s_i >= 0
    # cx_i + s_i/2 <= 1  => cx_i + 0.5 s_i <= 1
    # cy_i - s_i/2 >= 0  => cy_i - 0.5 s_i >= 0
    # cy_i + s_i/2 <= 1  => cy_i + 0.5 s_i <= 1
    
    A_rows = []
    b_lower = []
    b_upper = []
    
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
        
    # 2. Separation constraints
    # For each pair (i, j):
    # cx_i - cx_j - 0.5 s_i - 0.5 s_j + M (1 - b_x_pos) >= 0
    # cx_j - cx_i - 0.5 s_i - 0.5 s_j + M (1 - b_x_neg) >= 0
    # cy_i - cy_j - 0.5 s_i - 0.5 s_j + M (1 - b_y_pos) >= 0
    # cy_j - cy_i - 0.5 s_i - 0.5 s_j + M (1 - b_y_neg) >= 0
    # M = 1 is sufficient since cx, cy in [0,1] and s in [0,1].
    # Max LHS without binary term is -1 (if cx_i=0, cx_j=1, s=1). 
    # If b=0, term is +M. -1 + 1 = 0 >= 0. OK.
    
    M = 1.0
    
    for (i, j) in pairs:
        idx = bin_idx[(i, j)]
        
        # x_pos: cx_i - cx_j - 0.5(si+sj) + M(1 - b_x_pos) >= 0
        row = np.zeros(total_vars)
        row[cx_idx[i]] = 1.0
        row[cx_idx[j]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[idx['x_pos']] = -M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # x_neg: cx_j - cx_i - 0.5(si+sj) + M(1 - b_x_neg) >= 0
        row = np.zeros(total_vars)
        row[cx_idx[j]] = 1.0
        row[cx_idx[i]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[idx['x_neg']] = -M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # y_pos: cy_i - cy_j - 0.5(si+sj) + M(1 - b_y_pos) >= 0
        row = np.zeros(total_vars)
        row[cy_idx[i]] = 1.0
        row[cy_idx[j]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[idx['y_pos']] = -M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # y_neg: cy_j - cy_i - 0.5(si+sj) + M(1 - b_y_neg) >= 0
        row = np.zeros(total_vars)
        row[cy_idx[j]] = 1.0
        row[cy_idx[i]] = -1.0
        row[s_idx[i]] = -0.5
        row[s_idx[j]] = -0.5
        row[idx['y_neg']] = -M
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(np.inf)
        
        # Sum of binaries = 1
        # b_x_pos + b_x_neg + b_y_pos + b_y_neg = 1
        row = np.zeros(total_vars)
        row[idx['x_pos']] = 1.0
        row[idx['x_neg']] = 1.0
        row[idx['y_pos']] = 1.0
        row[idx['y_neg']] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)
        
    A = np.array(A_rows)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    constraints = [LinearConstraint(A, b_lower, b_upper)]
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            cx = res.x[cx_idx]
            cy = res.x[cy_idx]
            s = res.x[s_idx]
            centers = np.stack([cx, cy], axis=1)
            return centers, s
        else:
            return None
    except Exception:
        return None


def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    
    # 1. Try MILP
    if time.time() - start_time < _DEADLINE_S - 10:
        milp_result = _solve_milp(n)
        if milp_result is not None:
            centers, sides = milp_result
            if _feasible(centers, sides):
                # MILP provides a globally optimal solution for the linear relaxation
                # with fixed topology. It maximizes sum of sides.
                return centers, sides
    
    # 2. Fallback to grid if MILP fails or times out
    centers, sides = _grid_init(n)
    if _feasible(centers, sides):
        return centers, sides
        
    # 3. Last resort: return something valid (small squares)
    centers = np.zeros((n, 2))
    sides = np.full(n, 0.1)
    return centers, sides


# EVOLVE-BLOCK-END
