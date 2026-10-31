"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _objective(x):
    radii = x[2*N:]
    return -np.sum(radii)

def _boundary_constraints(x):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    return cons

def _overlap_constraints(x):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    # Vectorized
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    cons = (dist_sq - sum_r * sum_r)[mask]
    return cons

def _all_constraints(x):
    b_cons = _boundary_constraints(x)
    o_cons = _overlap_constraints(x)
    return np.concatenate([b_cons, o_cons])

def _soft_penalty_objective(x, beta=50.0):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    b_cons = _boundary_constraints(x)
    o_cons = _overlap_constraints(x)
    all_cons = np.concatenate([b_cons, o_cons])
    z = -beta * all_cons
    softplus_vals = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    penalty = np.sum(softplus_vals) / beta
    return -np.sum(radii) + penalty

def _is_valid(pos, radii, tol=1e-6):
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii, tol=1e-7):
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(50):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    if sum_r > 1e-9:
                        k = (dist - tol) / (sum_r + 1e-12)
                    else:
                        k = 0.99
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _qp_subproblem(pos, radii, active_pairs, active_walls, max_iter=200, tol=1e-12):
    """
    Solve a sequence of convex QP subproblems to improve the packing.
    We use scipy.optimize.linprog for the LP relaxation of the QP,
    or more practically, we use a custom projected gradient / active set method.
    
    Actually, for a true QP, we'd need a QP solver. Since we only have linprog (LP),
    we will use a "linearized" approach:
    1. Linearize the non-convex constraints at the current point.
    2. The objective is linear (sum of radii).
    3. The linearized constraints are linear.
    4. Solve the LP.
    5. Check feasibility of the original non-convex constraints.
    6. If feasible, update. If not, reduce step size (line search).
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Variables: x_i, y_i, r_i for i=0..N-1
    # Total variables: 3*N
    
    def build_lp(pos_curr, radii_curr, active_pairs, active_walls):
        c = np.zeros(3*N)
        # Maximize sum(r_i) -> Minimize -sum(r_i)
        c[2*N:] = -1.0
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints (always active in a sense, or we can just use bounds)
        # We'll use bounds for x, y, r
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x_i
            bounds.append((0.0, 1.0)) # y_i
            bounds.append((1e-6, 0.5)) # r_i
        
        # We need to enforce:
        # 1. x_i - r_i >= 0  =>  -x_i + r_i <= 0
        # 2. 1 - x_i - r_i >= 0  =>  x_i + r_i <= 1
        # 3. y_i - r_i >= 0  =>  -y_i + r_i <= 0
        # 4. 1 - y_i - r_i >= 0  =>  y_i + r_i <= 1
        
        for i in range(N):
            idx_x = 3*i
            idx_y = 3*i + 1
            idx_r = 3*i + 2
            
            # -x_i + r_i <= 0
            row = np.zeros(3*N)
            row[idx_x] = -1.0
            row[idx_r] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # x_i + r_i <= 1
            row = np.zeros(3*N)
            row[idx_x] = 1.0
            row[idx_r] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
            # -y_i + r_i <= 0
            row = np.zeros(3*N)
            row[idx_y] = -1.0
            row[idx_r] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # y_i + r_i <= 1
            row = np.zeros(3*N)
            row[idx_y] = 1.0
            row[idx_r] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
        # Pairwise constraints:
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Linearization around (pos_curr, radii_curr):
        # Let d_ij = pos_i - pos_j, r_sum = r_i + r_j
        # Constraint: ||d_ij||^2 - r_sum^2 >= 0
        # Gradient wrt pos_i: 2 * d_ij
        # Gradient wrt pos_j: -2 * d_ij
        # Gradient wrt r_i: -2 * r_sum
        # Gradient wrt r_j: -2 * r_sum
        # Linearized: 2*d_ij.x*(dx_i - dx_j) + 2*d_ij.y*(dy_i - dy_j) - 2*r_sum*(dr_i + dr_j) + (||d_ij||^2 - r_sum^2) >= 0
        # Note: dx_i, dy_i, etc are the *changes* from current pos/radii? No, linprog solves for absolute values.
        # So we substitute x_i = pos_curr_i + delta_x_i.
        # Actually, it's easier to just define the linear constraint on the absolute variables directly using the tangent plane.
        
        # Tangent plane at point P0:
        # f(x) >= 0
        # f(P0) + grad(f)(P0) . (x - P0) >= 0
        # grad(f)(P0) . x >= grad(f)(P0) . P0 - f(P0)
        
        for (i, j) in active_pairs:
            dx = pos_curr[i, 0] - pos_curr[j, 0]
            dy = pos_curr[i, 1] - pos_curr[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii_curr[i] + radii_curr[j]
            f0 = dist_sq - r_sum*r_sum
            
            # Gradients
            gx_i = 2 * dx
            gy_i = 2 * dy
            gx_j = -2 * dx
            gy_j = -2 * dy
            gr_i = -2 * r_sum
            gr_j = -2 * r_sum
            
            # Linearized constraint:
            # gx_i * x_i + gy_i * y_i + gr_i * r_i + gx_j * x_j + gy_j * y_j + gr_j * r_j >= C
            # where C = gx_i * pos_i + gy_i * pos_i + gr_i * r_i + ... - f0
            
            idx_x_i = 3*i
            idx_y_i = 3*i + 1
            idx_r_i = 3*i + 2
            idx_x_j = 3*j
            idx_y_j = 3*j + 1
            idx_r_j = 3*j + 2
            
            # We need A_ub @ x <= b_ub.
            # So we multiply by -1:
            # -gx_i * x_i - gy_i * y_i - gr_i * r_i - gx_j * x_j - gy_j * y_j - gr_j * r_j <= -C
            
            row = np.zeros(3*N)
            row[idx_x_i] = -gx_i
            row[idx_y_i] = -gy_i
            row[idx_r_i] = -gr_i
            row[idx_x_j] = -gx_j
            row[idx_y_j] = -gy_j
            row[idx_r_j] = -gr_j
            
            C = gx_i * pos_curr[i, 0] + gy_i * pos_curr[i, 1] + gr_i * radii_curr[i] + \
                gx_j * pos_curr[j, 0] + gy_j * pos_curr[j, 1] + gr_j * radii_curr[j] - f0
            
            A_ub.append(row)
            b_ub.append(-C)
            
        return np.array(c), np.array(A_ub).T, np.array(b_ub), bounds

    current_pos = pos
    current_radii = radii
    
    for it in range(max_iter):
        c, A_ub, b_ub, bounds = build_lp(current_pos, current_radii, active_pairs, active_walls)
        
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            break
            
        new_x = res.x
        new_pos = np.zeros((N, 2))
        new_radii = np.zeros(N)
        for i in range(N):
            new_pos[i, 0] = new_x[3*i]
            new_pos[i, 1] = new_x[3*i + 1]
            new_radii[i] = new_x[3*i + 2]
            
        # Check if the new solution is feasible for the *original* non-convex constraints
        new_pos_pp, new_radii_pp = _post_process(new_pos, new_radii, tol=1e-9)
        if _is_valid(new_pos_pp, new_radii_pp, tol=1e-8):
            # Check if it improved the objective
            if np.sum(new_radii_pp) > np.sum(current_radii) + 1e-10:
                current_pos = new_pos_pp
                current_radii = new_radii_pp
            else:
                # No improvement, stop
                break
        else:
            # Not feasible, we could try a line search, but for simplicity, stop
            # Or we could just break, as the LP solution is a relaxation
            break
            
    return current_pos, current_radii

def _get_active_constraints(pos, radii, tol=1e-5):
    active_pairs = []
    active_walls = []
    for i in range(N):
        if pos[i, 0] - radii[i] < tol:
            active_walls.append((i, 'left'))
        if 1.0 - pos[i, 0] - radii[i] < tol:
            active_walls.append((i, 'right'))
        if pos[i, 1] - radii[i] < tol:
            active_walls.append((i, 'bottom'))
        if 1.0 - pos[i, 1] - radii[i] < tol:
            active_walls.append((i, 'top'))
            
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            sum_r = radii[i] + radii[j]
            if dist_sq - sum_r*sum_r < tol:
                active_pairs.append((i, j))
                
    return active_pairs, active_walls

def _qp_optimize(pos, radii, time_budget=10.0):
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    # Initial active set
    active_pairs, active_walls = _get_active_constraints(pos, radii, tol=1e-4)
    
    # Run QP subproblem
    pos_new, radii_new = _qp_subproblem(pos, radii, active_pairs, active_walls)
    
    # Update active set and run again if time permits
    if time.time() - start_t < time_budget * 0.5:
        active_pairs, active_walls = _get_active_constraints(pos_new, radii_new, tol=1e-4)
        pos_new2, radii_new2 = _qp_subproblem(pos_new, radii_new, active_pairs, active_walls)
        if np.sum(radii_new2) > np.sum(radii_new):
            pos_new, radii_new = pos_new2, radii_new2
            
    return pos_new, radii_new

def _high_precision_polish(pos, radii, time_budget=5.0):
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, pos[i,0] - 0.1), min(1.0, pos[i,0] + 0.1)))
        bounds.append((max(0.0, pos[i,1] - 0.1), min(1.0, pos[i,1] + 0.1)))
    for i in range(N):
        bounds.append((max(1e-5, radii[i] * 0.9), min(0.5, radii[i] * 1.1)))
    
    def constraints_func(x):
        return _all_constraints(x)
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints_func},
                          options={'maxiter': 150, 'ftol': 1e-10})
        if result.success or result.fun < _objective(x0):
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-8)
            if _is_valid(pos_pp, radii_pp, tol=1e-7):
                pos, radii = pos_pp, radii_pp
    except Exception:
        pass
    
    if time.time() - start_t < time_budget * 0.5:
        try:
            x0 = np.concatenate([pos.flatten(), radii])
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 80, 'ftol': 1e-12})
            if result.success or result.fun < _objective(x0):
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-9)
                if _is_valid(pos_pp, radii_pp, tol=1e-8):
                    pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    if time.time() - start_t < time_budget * 0.8:
        try:
            x0 = np.concatenate([pos.flatten(), radii])
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 50, 'ftol': 1e-14})
            if result.success or result.fun < _objective(x0):
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-10)
                if _is_valid(pos_pp, radii_pp, tol=1e-9):
                    pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    return pos, radii

def _initial_greedy_best_fit():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(12000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            pos[k] = [x, y]
            radii[k] = r
            k += 1
    while k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.05
        k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _vectorized_refine(pos, radii, steps=500, step_size=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        current_step = step_size * (0.995 ** step)
        delta_pos = rng.normal(0, current_step, (N, 2))
        delta_radii = rng.normal(0, current_step * 0.3, N)
        new_pos = pos + delta_pos
        new_radii = radii + delta_radii
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        new_pos, new_radii = _post_process(new_pos, new_radii, tol=1e-8)
        if _is_valid(new_pos, new_radii, tol=1e-7):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    return best_pos, best_radii

def _smooth_optimize(pos, radii, beta_start=20.0, beta_end=100.0, n_iters=3):
    pos = pos.copy()
    radii = radii.copy()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for it in range(n_iters):
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        def obj(x):
            return _soft_penalty_objective(x, beta=beta)
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 150, 'ftol': 1e-12})
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-8)
            if _is_valid(pos_pp, radii_pp, tol=1e-7):
                pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
        lambda: _initial_random(6),
        lambda: _initial_random(7),
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    def constraints_func(x):
        return _all_constraints(x)

    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 25:
            break
            
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Step 1: Smooth penalty optimization
        try:
            pos_smooth, radii_smooth = _smooth_optimize(pos_init, rad_init, beta_start=20.0, beta_end=100.0, n_iters=3)
            score_smooth = np.sum(radii_smooth)
            if score_smooth > best_score:
                best_score = score_smooth
                best_pos = pos_smooth.copy()
                best_radii = radii_smooth.copy()
        except Exception:
            pass
        
        # Step 2: SLSQP with hard constraints
        if time.time() < deadline - 20:
            try:
                result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                                  constraints={'type': 'ineq', 'fun': constraints_func},
                                  options={'maxiter': 200, 'ftol': 1e-8})
                if result.success or result.fun < _objective(x0):
                    pos_opt = result.x[:2*N].reshape(N, 2)
                    radii_opt = result.x[2*N:]
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    if _is_valid(pos_pp, radii_pp):
                        score = np.sum(radii_pp)
                        if score > best_score:
                            best_score = score
                            best_pos = pos_pp.copy()
                            best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Step 3: QP Subproblem Optimization (New Strategy)
        if time.time() < deadline - 18 and best_pos is not None:
            try:
                pos_qp, radii_qp = _qp_optimize(best_pos, best_radii, time_budget=5.0)
                score_qp = np.sum(radii_qp)
                if score_qp > best_score:
                    best_score = score_qp
                    best_pos = pos_qp.copy()
                    best_radii = radii_qp.copy()
            except Exception:
                pass
        
        # Step 4: Vectorized refinement
        if time.time() < deadline - 12 and best_pos is not None:
            pos_ref, rad_ref = _vectorized_refine(best_pos, best_radii, steps=300, step_size=0.012, rng=np.random.RandomState(idx * 7 + 1))
            if _is_valid(pos_ref, rad_ref):
                score_ref = np.sum(rad_ref)
                if score_ref > best_score:
                    best_score = score_ref
                    best_pos = pos_ref.copy()
                    best_radii = rad_ref.copy()

    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 3:
        pos_polished, radii_polished = _high_precision_polish(best_pos, best_radii, time_budget=4.0)
        if _is_valid(pos_polished, radii_polished, tol=1e-8):
            score_polished = np.sum(radii_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pos = pos_polished.copy()
                best_radii = radii_polished.copy()

    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
