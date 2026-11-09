# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

def _is_valid(centers, sides, tol=1e-7):
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
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _soft_overlap_penalty(x, n=10, beta=50.0):
    """
    Smooth differentiable overlap penalty using soft-max of pairwise overlaps.
    x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]
    Returns (penalty_value, penalty_gradient)
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Compute pairwise overlaps smoothly
    # For each pair (i,j), overlap_x = max(0, t_x - |dx|), overlap_y = max(0, t_y - |dy|)
    # where t_x = (s_i + s_j)/2, t_y = (s_i + s_j)/2
    # We use a smooth approximation: soft_max(0, v) ≈ log(1+exp(beta*v))/beta for large beta
    
    penalty = 0.0
    grad = np.zeros(3*n)
    
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Left boundary: cx - s/2 < 0
        v = cx - s/2
        if v < 0:
            # smooth: v + log(1+exp(-beta*v))/beta
            # For simplicity, use quadratic
            penalty += 1e4 * v * v
            grad[i] += 1e4 * 2 * v
            grad[2*n+i] += 1e4 * 2 * v * (-0.5)
        # Right boundary: cx + s/2 > 1
        v = cx + s/2 - 1
        if v > 0:
            penalty += 1e4 * v * v
            grad[i] += 1e4 * 2 * v
            grad[2*n+i] += 1e4 * 2 * v * 0.5
        # Bottom boundary
        v = cy - s/2
        if v < 0:
            penalty += 1e4 * v * v
            grad[n+i] += 1e4 * 2 * v
            grad[2*n+i] += 1e4 * 2 * v * (-0.5)
        # Top boundary
        v = cy + s/2 - 1
        if v > 0:
            penalty += 1e4 * v * v
            grad[n+i] += 1e4 * 2 * v
            grad[2*n+i] += 1e4 * 2 * v * 0.5
    
    # Pairwise overlap penalties with smooth max
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0
            
            # Smooth overlap in x: max(0, t - |dx|)
            ax = t - abs(dx)
            # Smooth overlap in y: max(0, t - |dy|)
            ay = t - abs(dy)
            
            # Use softplus-like smoothing: f(v) = log(1+exp(beta*v))/beta
            # This is smooth and approximates max(0, v)
            def softplus(v, b):
                if v * b > 50:
                    return v  # linear regime
                elif v * b < -50:
                    return 0.0
                else:
                    return np.log1p(np.exp(b * v)) / b
            
            def softplus_grad(v, b):
                if v * b > 50:
                    return 1.0
                elif v * b < -50:
                    return 0.0
                else:
                    return 1.0 / (1.0 + np.exp(-b * v))
            
            ox = softplus(ax, beta)
            oy = softplus(ay, beta)
            
            # Overlap area approximation: ox * oy
            # Gradient with respect to positions and sizes
            gx_x = softplus_grad(ax, beta)  # d(ox)/d(ax)
            gy_y = softplus_grad(ay, beta)  # d(oy)/d(ay)
            
            # d(ax)/d(cx_i) = -sign(dx), d(ax)/d(cx_j) = sign(dx)
            # d(ax)/d(s_i) = 0.5, d(ax)/d(s_j) = 0.5
            # Similarly for y
            
            if dx >= 0:
                # ax = t - dx, d(ax)/d(cx_i) = -1, d(ax)/d(cx_j) = 1
                grad[i] += -oy * gx_x
                grad[j] += oy * gx_x
            else:
                # ax = t + dx, d(ax)/d(cx_i) = 1, d(ax)/d(cx_j) = -1
                grad[i] += oy * gx_x
                grad[j] += -oy * gx_x
            
            if dy >= 0:
                # ay = t - dy, d(ay)/d(cy_i) = -1, d(ay)/d(cy_j) = 1
                grad[n+i] += -ox * gy_y
                grad[n+j] += ox * gy_y
            else:
                # ay = t + dy, d(ay)/d(cy_i) = 1, d(ay)/d(cy_j) = -1
                grad[n+i] += ox * gy_y
                grad[n+j] += -ox * gy_y
            
            # d(ax)/d(s_i) = 0.5, d(ay)/d(s_i) = 0.5
            grad[2*n+i] += 0.5 * (oy * gx_x + ox * gy_y)
            grad[2*n+j] += 0.5 * (oy * gx_x + ox * gy_y)
            
            penalty += 100.0 * ox * oy
    
    return penalty, grad

def _soft_objective(x, n=10, beta=50.0):
    """
    Objective: -sum(sides) + soft_overlap_penalty
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    pen, _ = _soft_overlap_penalty(x, n, beta)
    return -np.sum(sides) + pen

def _soft_gradient(x, n=10, beta=50.0):
    """
    Gradient of soft objective.
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    _, pen_grad = _soft_overlap_penalty(x, n, beta)
    grad = pen_grad.copy()
    grad[2*n:] -= 1.0  # gradient of -sum(sides)
    return grad

def _l_bfgsb_optimize(x0, n=10, deadline=None, beta=50.0, maxiter=200):
    """
    Run L-BFGS-B on the soft-constraint problem.
    Returns (centers, sides, success)
    """
    # Bounds: centers in [0,1], sides in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
    for i in range(n):
        bounds.append((0.0, 1.0))  # cy
    for i in range(n):
        bounds.append((0.01, 0.5))  # sides
    
    try:
        res = minimize(
            fun=lambda x: _soft_objective(x, n, beta),
            x0=x0,
            jac=lambda x: _soft_gradient(x, n, beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e3:
            x_opt = res.x
            centers = x_opt[:2*n].reshape(n, 2)
            sides = x_opt[2*n:]
            
            # Clamp to ensure validity
            for i in range(n):
                s = max(0.01, min(0.5, sides[i]))
                sides[i] = s
                centers[i, 0] = max(s/2, min(1-s/2, centers[i, 0]))
                centers[i, 1] = max(s/2, min(1-s/2, centers[i, 1]))
            
            return centers, sides, True
        else:
            return None, None, False
    except Exception:
        return None, None, False

def _local_search(centers, sides, max_iters=2000, deadline=None, step_scale=0.02):
    """
    Local search with adaptive step sizes.
    """
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_score = np.sum(sides)
    
    rng = np.random.RandomState(123)
    step = step_scale
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(n)
        
        # Try perturbing position and/or size
        dcx = rng.uniform(-step, step)
        dcy = rng.uniform(-step, step)
        ds = rng.uniform(-step, step) * best_s[i]
        
        cx_new = best_c[i, 0] + dcx
        cy_new = best_c[i, 1] + dcy
        s_new = best_s[i] + ds
        s_new = max(0.01, min(0.5, s_new))
        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
        
        c_new = best_c.copy()
        s_new_arr = best_s.copy()
        c_new[i] = [cx_new, cy_new]
        s_new_arr[i] = s_new
        
        if _is_valid(c_new, s_new_arr):
            score = np.sum(s_new_arr)
            if score > best_score:
                best_score = score
                best_c = c_new
                best_s = s_new_arr
                step = min(step * 1.1, 0.1)  # increase step slightly
            else:
                step = max(step * 0.95, 0.001)  # decrease step
        else:
            step = max(step * 0.9, 0.001)
    
    return best_c, best_s

def _fixed_orientation_lp(centers, sides, n=10):
    """
    Given fixed side lengths, find optimal centers using LP with orientations
    determined by current relative positions.
    """
    num_pairs = n * (n - 1) // 2
    total_vars = 2 * n
    
    c_obj = np.zeros(total_vars)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        row = np.zeros(total_vars)
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[n+i] = -1.0
        A_ub.append(row)
        b_ub.append(-sides[i]/2.0)
        
        row = np.zeros(total_vars)
        row[n+i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - sides[i]/2.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            sep = (sides[i] + sides[j]) / 2.0
            
            if abs(dx) > abs(dy):
                # Horizontal separation
                if dx >= 0:
                    # i is to the right of j
                    row = np.zeros(total_vars)
                    row[i] = 1.0
                    row[j] = -1.0
                    A_ub.append(row)
                    b_ub.append(sep)
                else:
                    # i is to the left of j
                    row = np.zeros(total_vars)
                    row[i] = -1.0
                    row[j] = 1.0
                    A_ub.append(row)
                    b_ub.append(sep)
            else:
                # Vertical separation
                if dy >= 0:
                    # i is above j
                    row = np.zeros(total_vars)
                    row[n+i] = 1.0
                    row[n+j] = -1.0
                    A_ub.append(row)
                    b_ub.append(sep)
                else:
                    # i is below j
                    row = np.zeros(total_vars)
                    row[n+i] = -1.0
                    row[n+j] = 1.0
                    A_ub.append(row)
                    b_ub.append(sep)
    
    if not A_ub:
        return centers.copy(), True
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (2*n), method='highs')
    if res.status == 0:
        centers_out = res.x.reshape(n, 2)
        return centers_out, True
    return None, False

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # ---- Initial seeds ----
    seeds = []
    
    # Seed 1: 2x5 grid
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    seeds.append((c1, s1))
    
    # Seed 2: 5x2 grid
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        r = i // 2
        c = i % 2
        c2[i] = [(c + 0.5) * 0.5, (r + 0.5) * 0.2]
        s2[i] = 0.2
    seeds.append((c2, s2))
    
    # Seed 3: 4x4 grid (take 10)
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i // 4
        c = i % 4
        c3[i] = [(c + 0.5) * 0.25, (r + 0.5) * 0.25]
        s3[i] = 0.25
    seeds.append((c3, s3))
    
    # Seed 4: Mixed sizes
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    c4[0] = [0.25, 0.25]
    s4[0] = 0.4
    c4[1] = [0.75, 0.25]
    s4[1] = 0.4
    c4[2] = [0.25, 0.75]
    s4[2] = 0.3
    c4[3] = [0.75, 0.75]
    s4[3] = 0.3
    for i in range(6):
        r = i // 3
        c = i % 3
        c4[i+4] = [0.5 + (c-1)*0.15, 0.5 + (r-0.5)*0.15]
        s4[i+4] = 0.15
    seeds.append((c4, s4))
    
    # Update best from seeds
    for c_s, s_s in seeds:
        update_best(c_s, s_s)
    
    # ---- L-BFGS-B with soft constraints from multiple starts ----
    betas = [30.0, 50.0, 100.0]
    
    for c_s, s_s in seeds:
        if time.time() > deadline:
            break
        x0 = np.concatenate([c_s.flatten(), s_s])
        
        for beta in betas:
            if time.time() > deadline:
                break
            c_opt, s_opt, ok = _l_bfgsb_optimize(x0, n, deadline, beta, maxiter=150)
            if ok:
                update_best(c_opt, s_opt)
    
    # ---- Local search from best ----
    if best_centers is not None and time.time() < deadline:
        c_ls, s_ls = _local_search(best_centers, best_sides, max_iters=3000, deadline=deadline)
        update_best(c_ls, s_ls)
    
    # ---- Fixed orientation LP refinement ----
    if best_centers is not None and time.time() < deadline:
        c_cur = best_centers.copy()
        s_cur = best_sides.copy()
        for iteration in range(10):
            if time.time() > deadline:
                break
            c_fixed, ok = _fixed_orientation_lp(c_cur, s_cur, n)
            if ok and _is_valid(c_fixed, s_cur):
                c_cur = c_fixed
                # Try to increase sides
                s_try = s_cur * 1.01
                s_try = np.clip(s_try, 0.01, 0.5)
                c_try, ok2 = _fixed_orientation_lp(c_cur, s_try, n)
                if ok2 and _is_valid(c_try, s_try):
                    update_best(c_try, s_try)
                    c_cur = c_try
                    s_cur = s_try.copy()
                else:
                    # Try individual side increases
                    improved = False
                    for i in range(n):
                        if time.time() > deadline:
                            break
                        s_ind = s_cur.copy()
                        s_ind[i] = min(0.5, s_cur[i] * 1.02)
                        c_ind, ok3 = _fixed_orientation_lp(c_cur, s_ind, n)
                        if ok3 and _is_valid(c_ind, s_ind):
                            update_best(c_ind, s_ind)
                            c_cur = c_ind
                            s_cur = s_ind.copy()
                            improved = True
                    if not improved:
                        break
    
    # ---- More L-BFGS-B restarts from best ----
    if best_centers is not None and time.time() < deadline:
        x0 = np.concatenate([best_centers.flatten(), best_sides])
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline:
                break
            c_opt, s_opt, ok = _l_bfgsb_optimize(x0, n, deadline, beta, maxiter=200)
            if ok:
                update_best(c_opt, s_opt)
    
    # ---- Final local search polish ----
    if best_centers is not None and time.time() < deadline:
        c_final, s_final = _local_search(best_centers, best_sides, max_iters=3000, deadline=deadline)
        update_best(c_final, s_final)
    
    # ---- Return best found ----
    if best_centers is None:
        # Fallback: 2x5 grid
        best_centers = np.zeros((n, 2))
        best_sides = np.zeros(n)
        for i in range(n):
            r = i // 5
            c = i % 5
            best_centers[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
            best_sides[i] = 0.2
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
