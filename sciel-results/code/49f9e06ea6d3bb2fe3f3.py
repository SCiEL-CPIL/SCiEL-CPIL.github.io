# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def _check_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid (vectorized)."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Boundary check
    left = centers[:, 0] - sides / 2
    right = centers[:, 0] + sides / 2
    bottom = centers[:, 1] - sides / 2
    top = centers[:, 1] + sides / 2
    
    if np.any(left < -tol) or np.any(right > 1 + tol):
        return False
    if np.any(bottom < -tol) or np.any(top > 1 + tol):
        return False
        
    # Overlap check
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    
    t = (sides[:, None] + sides[None, :]) / 2 - tol
    
    iu = np.triu_indices(n, k=1)
    
    dx_ij = dx[iu]
    dy_ij = dy[iu]
    t_ij = t[iu]
    
    violations = (dx_ij < t_ij) & (dy_ij < t_ij)
    
    return not np.any(violations)

def _local_improve(centers, sides, tol=1e-7, max_iters=5000, step_size=0.01, time_limit=30, rng=None):
    """
    Local Improvement using Simulated Annealing.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    
    if rng is None:
        rng = np.random.default_rng(42)
        
    start_time = time.time()
    T = 0.05
    cooling = 0.998
    min_T = 1e-7
    current_val = best_val
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        idx = rng.integers(0, n)
        nc = best_centers.copy()
        ns = best_sides.copy()
        
        # Adaptive step size: larger when T is high
        noise = step_size * (0.5 + 0.5 * (T / 0.05))
        
        # Perturb center and side
        nc[idx, 0] += rng.uniform(-noise, noise)
        nc[idx, 1] += rng.uniform(-noise, noise)
        
        # Bias towards expanding the square if it's small, or moving it if it's large
        if ns[idx] < 0.3:
            ns[idx] += rng.uniform(-noise*0.5, noise)
        else:
            ns[idx] += rng.uniform(-noise, noise)
            
        ns[idx] = max(1e-6, min(1.0, ns[idx]))
        
        # Clip to bounds
        s = ns[idx]
        nc[idx, 0] = np.clip(nc[idx, 0], s/2, 1-s/2)
        nc[idx, 1] = np.clip(nc[idx, 1], s/2, 1-s/2)
        
        if _check_valid(nc, ns, tol):
            new_val = np.sum(ns)
            delta = new_val - current_val
            if delta > 0 or rng.random() < np.exp(delta / T):
                best_centers = nc
                best_sides = ns
                best_val = new_val
                current_val = new_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    T = max(min_T, T * 1.1)
                    no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                T = max(min_T, T * 1.1)
                no_improve_count = 0
                
        T *= cooling
        
    return best_centers, best_sides, best_val

def _heuristic_init_2_large_8_small():
    """2 large squares (0.5) + 8 small squares (0.25) layout."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.5 + (row + 0.5) * 0.25
        centers[2+k, 0] = cx
        centers[2+k, 1] = cy
        sides[2+k] = 0.25
    
    return centers, sides

def _heuristic_init_3_rows():
    """
    Try a 3-row layout:
    Row 1 (top): 4 squares of 0.25
    Row 2 (mid): 3 squares of 1/3
    Row 3 (bot): 3 squares of 1/3
    Sum: 4*0.25 + 6*(1/3) = 1 + 2 = 3.0
    """
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Top row: 4 squares of 0.25
    for k in range(4):
        centers[k, 0] = (k + 0.5) * 0.25
        centers[k, 1] = 0.875
        sides[k] = 0.25
        
    # Mid row: 3 squares of 1/3
    for k in range(3):
        centers[4+k, 0] = (k + 0.5) * (1/3)
        centers[4+k, 1] = 0.5
        sides[4+k] = 1/3
        
    # Bot row: 3 squares of 1/3
    for k in range(3):
        centers[7+k, 0] = (k + 0.5) * (1/3)
        centers[7+k, 1] = 1/6
        sides[7+k] = 1/3
        
    return centers, sides

def _solve_milp(n=10, tol=1e-7, time_limit=120, warm_start=None):
    """
    Solve the square packing problem using MILP.
    
    Variables:
    - cx_i, cy_i: center coordinates (continuous, [0, 1])
    - s_i: side length (continuous, [0, 1])
    - b_ij^L, b_ij^R, b_ij^B, b_ij^T: binary variables for separation direction
    
    For each pair (i, j), exactly one of the four separation constraints must hold:
    - L: cx_j - cx_i >= (s_i + s_j)/2  (i left of j)
    - R: cx_i - cx_j >= (s_i + s_j)/2  (i right of j)
    - B: cy_j - cy_i >= (s_i + s_j)/2  (i below j)
    - T: cy_i - cy_j >= (s_i + s_j)/2  (i above j)
    
    We use big-M constraints:
    cx_j - cx_i - (s_i + s_j)/2 + M*(1 - b_ij^L) >= -tol
    cx_i - cx_j - (s_i + s_j)/2 + M*(1 - b_ij^R) >= -tol
    cy_j - cy_i - (s_i + s_j)/2 + M*(1 - b_ij^B) >= -tol
    cy_i - cy_j - (s_i + s_j)/2 + M*(1 - b_ij^T) >= -tol
    b_ij^L + b_ij^R + b_ij^B + b_ij^T = 1
    """
    M = 2.0  # Big-M value (max possible distance is 1, so 2 is safe)
    
    # Number of pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(pairs)
    
    # Variable layout:
    # [cx_0, cy_0, s_0, ..., cx_{n-1}, cy_{n-1}, s_{n-1}, b_0^L, b_0^R, b_0^B, b_0^T, ...]
    # First 3*n continuous variables, then 4*num_pairs binary variables
    
    total_vars = 3 * n + 4 * num_pairs
    
    # Objective: minimize -sum(s_i)
    c = np.zeros(total_vars)
    for i in range(n):
        c[3*i + 2] = -1.0  # s_i
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.zeros(total_vars)
    
    for i in range(n):
        lb[3*i] = 0.0    # cx_i
        ub[3*i] = 1.0
        lb[3*i + 1] = 0.0  # cy_i
        ub[3*i + 1] = 1.0
        lb[3*i + 2] = 1e-8  # s_i
        ub[3*i + 2] = 1.0
    
    for k in range(num_pairs):
        base = 3 * n + 4 * k
        lb[base:base+4] = 0.0
        ub[base:base+4] = 1.0
    
    # Integrality
    integrality = np.zeros(total_vars)
    integrality[3*n:] = 1  # Binary variables
    
    # Constraints
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # Boundary constraints:
    # cx_i - s_i/2 >= 0  =>  cx_i - s_i/2 >= 0
    # 1 - cx_i - s_i/2 >= 0  =>  -cx_i - s_i/2 >= -1
    # cy_i - s_i/2 >= 0
    # 1 - cy_i - s_i/2 >= 0
    
    for i in range(n):
        # cx_i - s_i/2 >= 0
        row = np.zeros(total_vars)
        row[3*i] = 1.0
        row[3*i + 2] = -0.5
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)
        
        # -cx_i - s_i/2 >= -1
        row = np.zeros(total_vars)
        row[3*i] = -1.0
        row[3*i + 2] = -0.5
        A_rows.append(row)
        lb_rows.append(-1.0)
        ub_rows.append(np.inf)
        
        # cy_i - s_i/2 >= 0
        row = np.zeros(total_vars)
        row[3*i + 1] = 1.0
        row[3*i + 2] = -0.5
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)
        
        # -cy_i - s_i/2 >= -1
        row = np.zeros(total_vars)
        row[3*i + 1] = -1.0
        row[3*i + 2] = -0.5
        A_rows.append(row)
        lb_rows.append(-1.0)
        ub_rows.append(np.inf)
    
    # Non-overlap constraints for each pair
    for k, (i, j) in enumerate(pairs):
        base_b = 3 * n + 4 * k
        bL = base_b
        bR = base_b + 1
        bB = base_b + 2
        bT = base_b + 3
        
        # Sum of binaries = 1
        row = np.zeros(total_vars)
        row[bL] = 1.0
        row[bR] = 1.0
        row[bB] = 1.0
        row[bT] = 1.0
        A_rows.append(row)
        lb_rows.append(1.0)
        ub_rows.append(1.0)
        
        # L: cx_j - cx_i - (s_i + s_j)/2 + M*(1 - bL) >= -tol
        # => cx_j - cx_i - 0.5*s_i - 0.5*s_j + M - M*bL >= -tol
        row = np.zeros(total_vars)
        row[3*j] = 1.0
        row[3*i] = -1.0
        row[3*i + 2] = -0.5
        row[3*j + 2] = -0.5
        row[bL] = -M
        A_rows.append(row)
        lb_rows.append(-tol)
        ub_rows.append(np.inf)
        
        # R: cx_i - cx_j - (s_i + s_j)/2 + M*(1 - bR) >= -tol
        row = np.zeros(total_vars)
        row[3*i] = 1.0
        row[3*j] = -1.0
        row[3*i + 2] = -0.5
        row[3*j + 2] = -0.5
        row[bR] = -M
        A_rows.append(row)
        lb_rows.append(-tol)
        ub_rows.append(np.inf)
        
        # B: cy_j - cy_i - (s_i + s_j)/2 + M*(1 - bB) >= -tol
        row = np.zeros(total_vars)
        row[3*j + 1] = 1.0
        row[3*i + 1] = -1.0
        row[3*i + 2] = -0.5
        row[3*j + 2] = -0.5
        row[bB] = -M
        A_rows.append(row)
        lb_rows.append(-tol)
        ub_rows.append(np.inf)
        
        # T: cy_i - cy_j - (s_i + s_j)/2 + M*(1 - bT) >= -tol
        row = np.zeros(total_vars)
        row[3*i + 1] = 1.0
        row[3*j + 1] = -1.0
        row[3*i + 2] = -0.5
        row[3*j + 2] = -0.5
        row[bT] = -M
        A_rows.append(row)
        lb_rows.append(-tol)
        ub_rows.append(np.inf)
    
    A = np.array(A_rows)
    lb_arr = np.array(lb_rows)
    ub_arr = np.array(ub_rows)
    
    constraints = LinearConstraint(A, lb_arr, ub_arr)
    bounds = Bounds(lb, ub)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': time_limit, 'mip_rel_gap': 1e-6}
        )
        
        if result.success and result.x is not None:
            x = result.x
            cx = x[0::3][:n]
            cy = x[1::3][:n]
            s = x[2::3][:n]
            
            # Clip to valid range
            for i in range(n):
                s[i] = max(1e-8, min(1.0, s[i]))
                cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
                cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
            
            centers = np.column_stack([cx, cy])
            return centers, s, -result.fun
        
    except Exception as e:
        pass
    
    return None, None, None

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 620  # Safety margin for 630s limit
    
    best_centers = None
    best_sides = None
    best_val = -1.0
    
    def _try_config(centers, sides, label=""):
        nonlocal best_centers, best_sides, best_val
        if not _check_valid(centers, sides, tol):
            return
        val = np.sum(sides)
        if val > best_val:
            best_val = val
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # 1. Start with heuristic configurations
    inits = [
        ("2_large_8_small", _heuristic_init_2_large_8_small),
        ("3_rows", _heuristic_init_3_rows),
    ]
    
    for label, init_fn in inits:
        if time.time() - start_time > deadline - 60:
            break
        c, s = init_fn()
        _try_config(c, s, label)
        
    # Ensure we have a valid best
    if best_centers is None:
        c, s = _heuristic_init_2_large_8_small()
        best_centers = c
        best_sides = s
        best_val = np.sum(s)

    # 2. Try MILP with increasing time limits
    milp_time_limits = [60, 120, 180]
    for tl in milp_time_limits:
        if time.time() - start_time > deadline - 30:
            break
        
        elapsed_so_far = time.time() - start_time
        remaining = deadline - 30 - elapsed_so_far
        if remaining < 10:
            break
        
        actual_tl = min(tl, remaining)
        
        centers_milp, sides_milp, val_milp = _solve_milp(n, tol, time_limit=actual_tl)
        
        if centers_milp is not None:
            _try_config(centers_milp, sides_milp, f"milp_{tl}")
    
    # 3. Local improvement to polish the best solution
    if time.time() - start_time < deadline - 5:
        rng = np.random.default_rng(42)
        c_final, s_final, val_final = _local_improve(best_centers, best_sides, tol, max_iters=100000, step_size=0.001, time_limit=30, rng=rng)
        _try_config(c_final, s_final, "final_polish")
        
    # Fallback
    if best_centers is None:
        c, s = _heuristic_init_2_large_8_small()
        best_centers = c
        best_sides = s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
