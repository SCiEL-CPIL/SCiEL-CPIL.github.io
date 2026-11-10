# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, minimize

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600  # Safety margin before 630s kill
    
    # Try multiple approaches and keep the best valid result
    candidates = []
    
    # Approach 1: MILP
    if time.time() - start_time < deadline - 120:
        try:
            c1, s1 = _solve_milp(n, tol, time_limit=100)
            if c1 is not None and _check_valid(c1, s1, tol):
                candidates.append((c1, s1, np.sum(s1)))
        except Exception as e:
            print(f"MILP error: {e}")
    
    # Approach 2: Structured layouts with SLSQP optimization
    if time.time() - start_time < deadline - 60:
        layouts = _generate_structured_layouts(n, tol)
        for idx, (c0, s0, sep_dirs) in enumerate(layouts):
            if time.time() - start_time > deadline - 60:
                break
            try:
                c_opt, s_opt = _optimize_layout_slsqp(c0, s0, sep_dirs, n, tol, time_limit=30)
                if c_opt is not None and _check_valid(c_opt, s_opt, tol):
                    candidates.append((c_opt, s_opt, np.sum(s_opt)))
            except Exception as e:
                pass
    
    # Approach 3: Heuristic + aggressive local search
    if time.time() - start_time < deadline - 30:
        c_h, s_h = _heuristic_init()
        if _check_valid(c_h, s_h, tol):
            candidates.append((c_h, s_h, np.sum(s_h)))
    
    # Local polish on the best candidate
    if candidates:
        candidates.sort(key=lambda x: -x[2])
        best_c, best_s, best_val = candidates[0]
        
        # Aggressive local improvement
        if time.time() - start_time < deadline - 10:
            best_c, best_s, best_val = _local_improve_aggressive(
                best_c, best_s, tol, time_limit=int(deadline - time.time() - 10)
            )
        
        return best_c, best_s
    
    # Fallback: always return something valid
    c_f, s_f = _heuristic_init()
    return c_f, s_f


def _solve_milp(n, tol, time_limit=100):
    """Solve the MILP formulation."""
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(pairs)
    
    # Variable indices
    idx_x = np.arange(0, n)
    idx_y = np.arange(n, 2*n)
    idx_s = np.arange(2*n, 3*n)
    idx_u = np.arange(3*n, 3*n + num_pairs)
    idx_v = np.arange(3*n + num_pairs, 3*n + 2*num_pairs)
    idx_w = np.arange(3*n + 2*num_pairs, 3*n + 3*num_pairs)
    idx_z = np.arange(3*n + 3*num_pairs, 3*n + 4*num_pairs)
    
    total_vars = 3*n + 4*num_pairs
    
    # Objective: maximize sum(s)
    c = np.zeros(total_vars)
    c[idx_s] = -1.0
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Integrality for binary variables
    integrality = np.zeros(total_vars)
    integrality[idx_u] = 1
    integrality[idx_v] = 1
    integrality[idx_w] = 1
    integrality[idx_z] = 1
    
    # Constraints
    A_rows = []
    b_rows = []
    
    # Boundary constraints
    for i in range(n):
        # cx_i - s_i/2 >= 0
        row = np.zeros(total_vars)
        row[idx_x[i]] = -1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)
        
        # cx_i + s_i/2 <= 1
        row = np.zeros(total_vars)
        row[idx_x[i]] = 1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)
        
        # cy_i - s_i/2 >= 0
        row = np.zeros(total_vars)
        row[idx_y[i]] = -1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(0.0)
        
        # cy_i + s_i/2 <= 1
        row = np.zeros(total_vars)
        row[idx_y[i]] = 1.0
        row[idx_s[i]] = 0.5
        A_rows.append(row)
        b_rows.append(1.0)
    
    # Pair constraints
    M = 2.0  # Big M
    
    for k, (i, j) in enumerate(pairs):
        # u+v+w+z = 1
        row = np.zeros(total_vars)
        row[idx_u[k]] = 1.0
        row[idx_v[k]] = 1.0
        row[idx_w[k]] = 1.0
        row[idx_z[k]] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)
        
        row_neg = -row
        A_rows.append(row_neg)
        b_rows.append(-1.0)
        
        # Separation constraints
        # u: i left of j
        row = np.zeros(total_vars)
        row[idx_x[i]] = 1.0
        row[idx_x[j]] = -1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_u[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # v: j left of i
        row = np.zeros(total_vars)
        row[idx_x[i]] = -1.0
        row[idx_x[j]] = 1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_v[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # w: i below j
        row = np.zeros(total_vars)
        row[idx_y[i]] = 1.0
        row[idx_y[j]] = -1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_w[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
        
        # z: j below i
        row = np.zeros(total_vars)
        row[idx_y[i]] = -1.0
        row[idx_y[j]] = 1.0
        row[idx_s[i]] = 0.5
        row[idx_s[j]] = 0.5
        row[idx_z[k]] = -M
        A_rows.append(row)
        b_rows.append(0.0)
    
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, -np.inf, b),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": time_limit}
        )
        if res.success and res.x is not None:
            cx = res.x[idx_x]
            cy = res.x[idx_y]
            s = res.x[idx_s]
            centers = np.column_stack([cx, cy])
            return centers, s
        else:
            print(f"MILP status: {res.message}")
    except Exception as e:
        print(f"MILP exception: {e}")
    
    return None, None


def _generate_structured_layouts(n, tol):
    """Generate several structured layout topologies."""
    layouts = []
    
    # Layout 1: 2 large squares on bottom, 8 small in 2x4 grid on top
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # Large squares
    s[0] = 0.45
    s[1] = 0.45
    c[0] = [0.225, 0.225]
    c[1] = [0.775, 0.225]
    # Small squares in 2 rows x 4 cols
    small_s = 0.18
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.55 + row * 0.22 + 0.11
        c[2+k] = [cx, cy]
        s[2+k] = small_s
    # Separation directions: 0=i left of j, 1=j left of i, 2=i below j, 3=j below i
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 2: 3 columns of squares
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 3 columns, each with 3-4 squares
    col_widths = [0.33, 0.34, 0.33]
    x_starts = [0, 0.33, 0.67]
    # Column 0: 4 squares
    for k in range(4):
        c[k] = [0.165, (k+0.5)*0.25]
        s[k] = 0.24
    # Column 1: 3 squares
    for k in range(3):
        c[4+k] = [0.5, (k+0.5)*0.333]
        s[4+k] = 0.30
    # Column 2: 3 squares
    for k in range(3):
        c[7+k] = [0.835, (k+0.5)*0.333]
        s[7+k] = 0.30
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 3: 4 large + 6 small
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 4 large in corners
    large_s = 0.35
    c[0] = [0.175, 0.175]
    c[1] = [0.825, 0.175]
    c[2] = [0.175, 0.825]
    c[3] = [0.825, 0.825]
    s[:4] = large_s
    # 6 small in center
    small_s = 0.15
    positions = [
        [0.35, 0.35], [0.5, 0.35], [0.65, 0.35],
        [0.35, 0.65], [0.5, 0.65], [0.65, 0.65]
    ]
    for k in range(6):
        c[4+k] = positions[k]
        s[4+k] = small_s
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 4: 5 large + 5 small
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 5 large
    large_s = 0.30
    c[0] = [0.15, 0.15]
    c[1] = [0.5, 0.15]
    c[2] = [0.85, 0.15]
    c[3] = [0.15, 0.5]
    c[4] = [0.85, 0.5]
    s[:5] = large_s
    # 5 small
    small_s = 0.18
    c[5] = [0.5, 0.5]
    c[6] = [0.3, 0.8]
    c[7] = [0.7, 0.8]
    c[8] = [0.5, 0.85]
    c[9] = [0.15, 0.85]
    s[5:] = small_s
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    return layouts


def _optimize_layout_slsqp(c0, s0, sep_dirs, n, tol, time_limit=30):
    """Optimize a layout with fixed separation directions using SLSQP."""
    # Variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = c0[i, 0]
        x0[3*i+1] = c0[i, 1]
        x0[3*i+2] = s0[i]
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 1))
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i]) - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i+1] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i+1]) - x[3*i+2] - tol})
    
    # Separation constraints based on sep_dirs
    for (i, j), direction in sep_dirs.items():
        if direction == 0:  # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*j] - x[3*i]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 1:  # j left of i: cx_j + s_j/2 <= cx_i - s_i/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*i] - x[3*j]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 2:  # i below j: cy_i + s_i/2 <= cy_j - s_j/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*j+1] - x[3*i+1]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 3:  # j below i: cy_j + s_j/2 <= cy_i - s_i/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*i+1] - x[3*j+1]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
    
    def objective(x):
        return -np.sum(x[2::3])
    
    start_time = time.time()
    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10}
        )
        
        if time.time() - start_time > time_limit:
            return None, None
        
        if result.fun is not None and np.isfinite(result.fun):
            x_opt = result.x
            centers = np.column_stack([x_opt[0::3], x_opt[1::3]])
            sides = x_opt[2::3]
            return centers, sides
    except Exception:
        pass
    
    return None, None


def _check_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
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
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_sep = (sides[i] + sides[j])/2 - tol
            if dx < min_sep and dy < min_sep:
                return False
    return True


def _heuristic_init():
    """Generate a valid heuristic initial configuration."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 2 large squares
    centers[0] = [0.25, 0.25]
    sides[0] = 0.45
    centers[1] = [0.75, 0.25]
    sides[1] = 0.45
    
    # 8 small squares in 2x4 grid
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.55 + row * 0.22 + 0.11
        centers[2+k, 0] = cx
        centers[2+k, 1] = cy
        sides[2+k] = 0.18
    
    return centers, sides


def _local_improve_aggressive(centers, sides, tol=1e-7, time_limit=60):
    """Aggressive local improvement: try to grow each square, then fine-tune positions."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    
    start_time = time.time()
    
    # Phase 1: Try to grow each square as much as possible
    for round_num in range(50):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            # Try to increase side length
            max_s = 1.0
            # Binary search for max feasible side
            lo, hi = best_sides[i], max_s
            for _ in range(20):
                mid = (lo + hi) / 2
                if _can_grow_square(best_centers, best_sides, i, mid, tol):
                    lo = mid
                else:
                    hi = mid
            
            if lo > best_sides[i] + 1e-8:
                best_sides[i] = lo
                best_val = np.sum(best_sides)
                improved = True
        
        if not improved:
            break
    
    # Phase 2: Fine-tune positions with small perturbations
    rng = np.random.default_rng(42)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() - start_time > time_limit:
            break
        
        for it in range(200):
            if time.time() - start_time > time_limit:
                break
            
            idx = rng.integers(0, n)
            nc = best_centers.copy()
            ns = best_sides.copy()
            
            nc[idx, 0] += rng.uniform(-step, step)
            nc[idx, 1] += rng.uniform(-step, step)
            
            # Clip to valid range
            s = ns[idx]
            nc[idx, 0] = np.clip(nc[idx, 0], s/2, 1-s/2)
            nc[idx, 1] = np.clip(nc[idx, 1], s/2, 1-s/2)
            
            # Check validity
            valid = True
            for j in range(n):
                if j == idx:
                    continue
                dx = abs(nc[idx, 0] - best_centers[j, 0])
                dy = abs(nc[idx, 1] - best_centers[j, 1])
                min_sep = (ns[idx] + best_sides[j])/2 - tol
                if dx < min_sep and dy < min_sep:
                    valid = False
                    break
            
            if valid:
                best_centers = nc
                best_val = np.sum(best_sides)
    
    return best_centers, best_sides, best_val


def _can_grow_square(centers, sides, idx, new_s, tol):
    """Check if we can grow square idx to new_s without violating constraints."""
    n = len(sides)
    cx = centers[idx, 0]
    cy = centers[idx, 1]
    
    # Check boundary
    if cx - new_s/2 < -tol or cx + new_s/2 > 1 + tol:
        return False
    if cy - new_s/2 < -tol or cy + new_s/2 > 1 + tol:
        return False
    
    # Check overlap with others
    for j in range(n):
        if j == idx:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        min_sep = (new_s + sides[j])/2 - tol
        if dx < min_sep and dy < min_sep:
            return False
    
    return True
# EVOLVE-BLOCK-END
