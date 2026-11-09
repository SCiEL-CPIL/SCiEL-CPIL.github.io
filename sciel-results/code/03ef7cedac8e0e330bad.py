# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize

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
    x_i, y_i (continuous, [0, 1])
    s_i (continuous, [0, 1])
    b_{ij}^k (binary, k=1..4) for i < j
    """
    # Variable ordering:
    # 0..n-1: x_i
    # n..2n-1: y_i
    # 2n..3n-1: s_i
    # 3n..: binaries for pairs
    
    num_pairs = n * (n - 1) // 2
    num_binaries = num_pairs * 4
    total_vars = 3 * n + num_binaries
    
    # Objective: Maximize sum(s_i) -> Minimize -sum(s_i)
    c = np.zeros(total_vars)
    c[2*n : 3*n] = -1.0
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    # s_i > 0 is handled by lb=0, but we might want a small epsilon to avoid 0 size if it helps stability? 
    # Actually, s_i=0 is feasible but suboptimal. The solver will maximize.
    # However, binary variables must be 0 or 1.
    lb[3*n:] = 0
    ub[3*n:] = 1
    
    # Integrality
    integrality = np.zeros(total_vars)
    integrality[3*n:] = 1  # binaries are integer
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # 1. Boundary Constraints
    # x_i - s_i/2 >= 0  =>  -x_i + 0.5*s_i <= 0
    # x_i + s_i/2 <= 1  =>   x_i + 0.5*s_i <= 1
    # y_i - s_i/2 >= 0  =>  -y_i + 0.5*s_i <= 0
    # y_i + s_i/2 <= 1  =>   y_i + 0.5*s_i <= 1
    
    for i in range(n):
        # -x_i + 0.5 s_i <= 0
        row = np.zeros(total_vars)
        row[i] = -1.0
        row[2*n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[i] = 1.0
        row[2*n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -y_i + 0.5 s_i <= 0
        row = np.zeros(total_vars)
        row[n + i] = -1.0
        row[2*n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i + 0.5 s_i <= 1
        row = np.zeros(total_vars)
        row[n + i] = 1.0
        row[2*n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # 2. Non-overlap constraints
    # For each pair (i, j) with i < j:
    # We have 4 binaries: b1 (i left of j), b2 (i right of j), b3 (i below j), b4 (i above j)
    # Constraint sum(b) = 1 is an equality constraint.
    # Separation constraints:
    # x_i - x_j >= (s_i + s_j)/2 - M * (1 - b1)
    # x_j - x_i >= (s_i + s_j)/2 - M * (1 - b2)
    # y_i - y_j >= (s_i + s_j)/2 - M * (1 - b3)
    # y_j - y_i >= (s_i + s_j)/2 - M * (1 - b4)
    
    # M should be large enough. Since x,y in [0,1], M=1 is sufficient.
    M = 1.0
    
    # We will build A_eq, b_eq for the sum(b)=1 constraints
    # And A_ub, b_ub for the separation constraints
    
    # Let's use LinearConstraint which can handle both ub and lb, or separate lists.
    # Easier to build A_ub and b_ub for inequalities and A_eq, b_eq for equalities.
    
    A_ub_list = A_ub
    b_ub_list = b_ub
    
    bin_idx = 3 * n
    pair_idx = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            b1_idx = bin_idx + pair_idx * 4
            b2_idx = b1_idx + 1
            b3_idx = b1_idx + 2
            b4_idx = b1_idx + 3
            
            # 1. x_i - x_j - (s_i+s_j)/2 + M*b1 >= 0
            # => -x_i + x_j + 0.5*s_i + 0.5*s_j - M*b1 <= 0
            row = np.zeros(total_vars)
            row[i] = -1.0
            row[j] = 1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[b1_idx] = -M
            A_ub_list.append(row)
            b_ub_list.append(0.0)
            
            # 2. x_j - x_i - (s_i+s_j)/2 + M*b2 >= 0
            # => x_i - x_j + 0.5*s_i + 0.5*s_j - M*b2 <= 0
            row = np.zeros(total_vars)
            row[i] = 1.0
            row[j] = -1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[b2_idx] = -M
            A_ub_list.append(row)
            b_ub_list.append(0.0)
            
            # 3. y_i - y_j - (s_i+s_j)/2 + M*b3 >= 0
            # => -y_i + y_j + 0.5*s_i + 0.5*s_j - M*b3 <= 0
            row = np.zeros(total_vars)
            row[n + i] = -1.0
            row[n + j] = 1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[b3_idx] = -M
            A_ub_list.append(row)
            b_ub_list.append(0.0)
            
            # 4. y_j - y_i - (s_i+s_j)/2 + M*b4 >= 0
            # => y_i - y_j + 0.5*s_i + 0.5*s_j - M*b4 <= 0
            row = np.zeros(total_vars)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[2*n + i] = 0.5
            row[2*n + j] = 0.5
            row[b4_idx] = -M
            A_ub_list.append(row)
            b_ub_list.append(0.0)
            
            pair_idx += 1
            
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    # Equality constraints: sum of 4 binaries for each pair = 1
    A_eq_list = []
    b_eq_list = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(total_vars)
            base = 3 * n + pair_idx * 4
            row[base:base+4] = 1.0
            A_eq_list.append(row)
            b_eq_list.append(1.0)
            pair_idx += 1
            
    A_eq = np.array(A_eq_list)
    b_eq = np.array(b_eq_list)
    
    # Combine constraints
    # LinearConstraint expects A, lb, ub
    # Inequalities: A_ub @ x <= b_ub  => A_ub @ x >= -inf, A_ub @ x <= b_ub
    # Equalities: A_eq @ x = b_eq => A_eq @ x >= b_eq, A_eq @ x <= b_eq
    
    A_total = np.vstack([A_ub, A_eq])
    lb_total = np.concatenate([np.full(A_ub.shape[0], -np.inf), b_eq])
    ub_total = np.concatenate([b_ub, b_eq])
    
    constraints = LinearConstraint(A_total, lb_total, ub_total)
    
    # Bounds for x, y, s are [0, 1]
    # Bounds for binaries are [0, 1]
    # lb and ub already set above
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
        if res.success:
            x = res.x
            cx = x[:n]
            cy = x[n:2*n]
            s = x[2*n:3*n]
            centers = np.stack([cx, cy], axis=1)
            sides = s
            return centers, sides
        else:
            return None, None
    except Exception as e:
        return None, None

def _local_optimize(centers, sides, n=10, max_iter=1000):
    """
    Local optimization using SLSQP with penalty for overlaps.
    """
    def objective(v):
        cx = v[:n]
        cy = v[n:2*n]
        s = v[2*n:3*n]
        # Penalty for going out of bounds
        pen = 0.0
        for i in range(n):
            if cx[i] - s[i]/2 < 0: pen += (cx[i] - s[i]/2)**2
            if cx[i] + s[i]/2 > 1: pen += (cx[i] + s[i]/2 - 1)**2
            if cy[i] - s[i]/2 < 0: pen += (cy[i] - s[i]/2)**2
            if cy[i] + s[i]/2 > 1: pen += (cy[i] + s[i]/2 - 1)**2
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                if dx < req_x and dy < req_y:
                    # They overlap
                    # Push them apart
                    pen += (req_x - dx)**2 + (req_y - dy)**2
        return -np.sum(s) + 100.0 * pen

    v0 = np.concatenate([centers.flatten(), sides])
    
    bounds = [(0, 1)] * (2*n) + [(1e-6, 1)] * n
    
    res = minimize(objective, v0, method='SLSQP', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12})
    
    if res.success:
        cx = res.x[:n]
        cy = res.x[n:2*n]
        s = res.x[2*n:3*n]
        return np.stack([cx, cy], axis=1), s
    return None, None

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    
    best_centers = None
    best_sides = None
    best_score = -np.inf

    # 1. Try MILP
    if time.time() - start_time < _DEADLINE_S - 30:
        centers_milp, sides_milp = _solve_milp(n)
        if centers_milp is not None and _feasible(centers_milp, sides_milp):
            best_centers = centers_milp
            best_sides = sides_milp
            best_score = np.sum(sides_milp)

    # 2. Try Local Optimization from Grid
    centers_grid, sides_grid = _grid_init(n)
    if _feasible(centers_grid, sides_grid):
        score_grid = np.sum(sides_grid)
        if best_score < score_grid:
            best_centers = centers_grid
            best_sides = sides_grid
            best_score = score_grid

    # 3. Polish with Local Optimization
    if time.time() - start_time < _DEADLINE_S - 10:
        # Use the best found so far as initial guess
        init_c, init_s = best_centers, best_sides
        centers_loc, sides_loc = _local_optimize(init_c, init_s, n)
        if centers_loc is not None and _feasible(centers_loc, sides_loc):
            score_loc = np.sum(sides_loc)
            if score_loc > best_score:
                best_centers = centers_loc
                best_sides = sides_loc
                best_score = score_loc

    # 4. Emergency fallback
    if best_centers is None:
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.1)
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
