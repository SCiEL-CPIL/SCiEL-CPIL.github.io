"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _objective(x, w, h):
    """
    Objective: maximize sum of radii.
    x: [x_0..x_{n-1}, y_0..y_{n-1}]
    w, h: rectangle dimensions (fixed during optimization)
    Returns: negative sum of radii (to minimize)
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    
    # Compute all constraints for each circle
    # Constraint types:
    # 1. cx[i] >= r_i  -> r_i <= cx[i]
    # 2. w - cx[i] >= r_i -> r_i <= w - cx[i]
    # 3. cy[i] >= r_i -> r_i <= cy[i]
    # 4. h - cy[i] >= r_i -> r_i <= h - cy[i]
    # 5. For each pair (i,j): dist(i,j) >= r_i + r_j
    
    # We want to maximize sum(r_i) subject to:
    # r_i <= min(cx[i], w-cx[i], cy[i], h-cy[i])
    # r_i + r_j <= dist(i,j) for all i<j
    
    # This is a linear program in r_i!
    # Maximize sum(r_i)
    # Subject to:
    #   r_i <= wall_limit_i  (4n constraints)
    #   r_i + r_j <= dist_ij (n*(n-1)/2 constraints)
    #   r_i >= 0 (n constraints)
    
    # Let's use scipy's linprog or just solve it directly
    # Actually, for n=21, this is 21 variables and ~21*4 + 21*20/2 = 84 + 210 = 294 constraints
    # This is very fast!
    
    # Wall limits
    wall_limits = np.minimum(np.minimum(cx, w - cx), np.minimum(cy, h - cy))
    
    # Pairwise distances
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    
    # We need to solve:
    # max sum(r)
    # s.t. r_i <= wall_limits[i] for all i
    #      r_i + r_j <= D[i,j] for all i < j
    #      r_i >= 0
    
    # This can be solved as a linear program. Let's use scipy.optimize.linprog
    # But linprog minimizes, so we minimize -sum(r)
    
    # Number of variables: n
    # Number of constraints: n (upper bounds) + n*(n-1)/2 (pairwise)
    
    # Actually, let's just use a simple approach: 
    # The optimal r_i is the minimum of all constraints that involve r_i
    # But constraints are coupled: r_i + r_j <= D[i,j]
    
    # This is a standard LP. Let's solve it with linprog.
    
    # c = [-1, -1, ..., -1] (minimize -sum(r))
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # 1. r_i <= wall_limits[i]:  e_i @ r <= wall_limits[i]
    # 2. r_i + r_j <= D[i,j]
    
    # Let's build A_ub and b_ub
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n + n_pairs, n))
    b_ub = np.zeros(n + n_pairs)
    
    # Wall constraints
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = wall_limits[i]
    
    # Pairwise constraints
    idx = n
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = D[i, j]
            idx += 1
    
    # Bounds: 0 <= r_i <= wall_limits[i]
    bounds = [(0.0, wall_limits[i]) for i in range(n)]
    
    try:
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun  # We minimized -sum(r), so return sum(r)
        else:
            # Fallback: use wall limits
            return np.sum(np.maximum(wall_limits, 0))
    except Exception:
        # Fallback: use wall limits
        return np.sum(np.maximum(wall_limits, 0))

def _optimize_positions(w, h, x0=None, time_limit=5.0):
    """
    Optimize circle positions for fixed rectangle dimensions w, h.
    Uses L-BFGS-B on the smooth objective (sum of radii from LP).
    """
    n = N
    
    if x0 is None:
        # Grid initialization
        n_cols = int(math.ceil(math.sqrt(n * w / h)))
        n_rows = int(math.ceil(n / n_cols))
        dx = w / n_cols
        dy = h / n_rows
        x0 = np.zeros(2*n)
        for i in range(n):
            c = i % n_cols
            r = i // n_cols
            x0[i] = (c + 0.5) * dx
            x0[n+i] = (r + 0.5) * dy
    
    # Bounds for positions
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
    for i in range(n):
        bounds.append((0.0, h))
    
    def obj(x):
        return -_objective(x, w, h)
    
    start = time.time()
    try:
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30,
                                  'maxfun': 200})
        return result.x
    except Exception:
        return x0

def _full_optimize(w0, h0, x0_pos=None, time_limit=15.0):
    """
    Jointly optimize positions and rectangle dimensions.
    """
    n = N
    
    # x = [positions (2n), w]
    # h = 2 - w
    
    if x0_pos is None:
        x0_pos = np.zeros(2*n)
        n_cols = int(math.ceil(math.sqrt(n * w0 / h0)))
        n_rows = int(math.ceil(n / n_cols))
        dx = w0 / n_cols
        dy = h0 / n_rows
        for i in range(n):
            c = i % n_cols
            r = i // n_cols
            x0_pos[i] = (c + 0.5) * dx
            x0_pos[n+i] = (r + 0.5) * dy
    
    x0 = np.concatenate([x0_pos, [w0]])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.9))
    for i in range(n):
        bounds.append((0.0, 1.9))
    bounds.append((0.2, 1.8))
    
    def obj(x):
        pos = x[:2*n]
        w = x[2*n]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return 1e10
        return -_objective(pos, w, h)
    
    start = time.time()
    try:
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30,
                                  'maxfun': 300})
        return result.x
    except Exception:
        return x0

def _simulated_annealing(w, h, x0, iterations=5000, time_limit=10.0):
    """
    Simulated annealing for position optimization with fixed w, h.
    """
    n = N
    cx = x0[:n].copy()
    cy = x0[n:2*n].copy()
    
    def eval_pos(cx, cy):
        return _objective(np.concatenate([cx, cy]), w, h)
    
    current_val = eval_pos(cx, cy)
    best_val = current_val
    best_cx = cx.copy()
    best_cy = cy.copy()
    
    T = 0.05
    T_min = 0.0001
    cooling = (T_min / T) ** (1.0 / iterations)
    
    start = time.time()
    for it in range(iterations):
        if time.time() - start > time_limit:
            break
        
        # Random perturbation
        idx = np.random.randint(n)
        delta = np.random.uniform(-0.05, 0.05)
        if np.random.rand() < 0.5:
            cx_new = cx.copy()
            cx_new[idx] = np.clip(cx_new[idx] + delta, 0, w)
            cy_new = cy
        else:
            cx_new = cx
            cy_new = cy.copy()
            cy_new[idx] = np.clip(cy_new[idx] + delta, 0, h)
        
        new_val = eval_pos(cx_new, cy_new)
        delta_val = new_val - current_val
        
        if delta_val > 0 or np.random.rand() < math.exp(delta_val / T):
            cx = cx_new
            cy = cy_new
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_cx = cx.copy()
                best_cy = cy.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return np.concatenate([best_cx, best_cy]), best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s
    n = N
    
    best_val = -1e10
    best_result = None
    
    # Try multiple aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1, 0.5, 1.5]
    
    # Phase 1: Initial optimization for each aspect
    for aspect in aspects:
        if time.time() > deadline - 30:
            break
        
        w0 = 2.0 * aspect / (1.0 + aspect)
        h0 = 2.0 / (1.0 + aspect)
        
        # Optimize positions for fixed aspect
        x_pos = _optimize_positions(w0, h0, time_limit=5.0)
        val = _objective(x_pos, w0, h0)
        
        if val > best_val:
            best_val = val
            best_result = (x_pos.copy(), w0, h0)
    
    # Phase 2: Joint optimization starting from best
    if best_result is not None and time.time() < deadline - 20:
        x_pos, w0, h0 = best_result
        x_full = _full_optimize(w0, h0, x_pos, time_limit=15.0)
        pos = x_full[:2*n]
        w = x_full[2*n]
        h = 2.0 - w
        val = _objective(pos, w, h)
        if val > best_val:
            best_val = val
            best_result = (pos.copy(), w, h)
    
    # Phase 3: Simulated annealing refinement
    if best_result is not None and time.time() < deadline - 10:
        x_pos, w, h = best_result
        x_sa, val_sa = _simulated_annealing(w, h, x_pos, iterations=3000, time_limit=8.0)
        if val_sa > best_val:
            best_val = val_sa
            best_result = (x_sa.copy(), w, h)
    
    # Phase 4: Final polish with another optimization round
    if best_result is not None and time.time() < deadline - 5:
        x_pos, w, h = best_result
        x_pos2 = _optimize_positions(w, h, x_pos, time_limit=3.0)
        val2 = _objective(x_pos2, w, h)
        if val2 > best_val:
            best_val = val2
            best_result = (x_pos2.copy(), w, h)
    
    # Build output
    if best_result is None:
        # Fallback
        W = 1.0
        H = 1.0
        n_cols = int(math.ceil(math.sqrt(n * W / H)))
        dx = W / n_cols
        dy = H / n_cols
        cx = np.array([(i % n_cols + 0.5) * dx for i in range(n)])
        cy = np.array([(i // n_cols + 0.5) * dy for i in range(n)])
        radii = np.full(n, 0.5 * min(dx, dy) - 0.005)
        return np.column_stack([cx, cy, radii])
    
    cx = best_result[0][:n]
    cy = best_result[0][n:2*n]
    w = best_result[1]
    h = best_result[2]
    
    # Compute final radii using LP
    wall_limits = np.minimum(np.minimum(cx, w - cx), np.minimum(cy, h - cy))
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n + n_pairs, n))
    b_ub = np.zeros(n + n_pairs)
    
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = wall_limits[i]
    
    idx = n
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = D[i, j]
            idx += 1
    
    bounds = [(0.0, wall_limits[i]) for i in range(n)]
    
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
        else:
            radii = np.maximum(wall_limits, 0)
    except Exception:
        radii = np.maximum(wall_limits, 0)
    
    radii = np.maximum(radii, 0.0)
    
    # Clamp positions
    for i in range(n):
        r = radii[i]
        cx[i] = np.clip(cx[i], r, w - r)
        cy[i] = np.clip(cy[i], r, h - r)
    
    # Recompute radii after clamp
    wall_limits = np.minimum(np.minimum(cx, w - cx), np.minimum(cy, h - cy))
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    D = np.sqrt(dx*dx + dy*dy)
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n + n_pairs, n))
    b_ub = np.zeros(n + n_pairs)
    
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = wall_limits[i]
    
    idx = n
    for i in range(n):
        for j in range(i+1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = D[i, j]
            idx += 1
    
    bounds = [(0.0, wall_limits[i]) for i in range(n)]
    
    try:
        from scipy.optimize import linprog
        c = -np.ones(n)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
        else:
            radii = np.maximum(wall_limits, 0)
    except Exception:
        radii = np.maximum(wall_limits, 0)
    
    radii = np.maximum(radii, 0.0)
    
    return np.column_stack([cx, cy, radii])

# EVOLVE-BLOCK-END
