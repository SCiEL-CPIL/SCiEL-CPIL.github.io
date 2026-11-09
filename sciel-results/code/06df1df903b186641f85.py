import numpy as np
import time
from scipy.optimize import linprog

N = 26
_DEADLINE = 145.0  # Safety margin from 150s limit

def is_valid(x, y, r, tol=1e-6):
    """
    Checks if the configuration is valid:
    1. All radii are non-negative.
    2. All circles are inside the unit square.
    3. No circles overlap.
    """
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    # Check pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    # Check for overlaps with tolerance
    if np.any(dist < min_dist - tol):
        return False
    return True

def solve_radii_lp(x, y, epsilon=1e-9):
    """
    Solves the Linear Programming problem to maximize sum(r)
    given fixed centers x, y.
    """
    n = len(x)
    c = -np.ones(n)
    
    A_rows = []
    b_vals = []
    
    # Pairwise constraints: r_i + r_j <= d_ij - epsilon
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < epsilon:
                return None
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_vals.append(d - epsilon)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i) - epsilon
    b_i = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_vals.append(b_i[i] - epsilon)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = res.x
            # Verify validity
            if is_valid(x, y, r):
                return r
    except Exception:
        pass
        
    return None

def objective_and_grad(z, penalty_weight):
    """
    Objective function and gradient for L-BFGS-B.
    z is a vector of length 3*N: [x_0, ..., x_N-1, y_0, ..., y_N-1, r_0, ..., r_N-1]
    We want to maximize sum(r), so we minimize -sum(r) + penalty(violations).
    """
    N_local = len(z) // 3
    x = z[:N_local]
    y = z[N_local:2*N_local]
    r = z[2*N_local:]
    
    # We want to maximize sum(r), so objective is -sum(r)
    obj = -np.sum(r)
    grad_x = np.zeros(N_local)
    grad_y = np.zeros(N_local)
    grad_r = -np.ones(N_local)
    
    # --- Overlap Penalty ---
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero in gradient calculation
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    # Penalty term: penalty_weight * sum(overlap^2)
    obj += penalty_weight * np.sum(overlap**2)
    
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    grad_x += -2.0 * penalty_weight * np.sum(overlap * ux, axis=1)
    grad_y += -2.0 * penalty_weight * np.sum(overlap * uy, axis=1)
    
    grad_r += 2.0 * penalty_weight * np.sum(overlap, axis=1)
    
    # --- Boundary Penalty ---
    vx1 = np.maximum(0, r - x)
    vx2 = np.maximum(0, r - (1 - x))
    vy1 = np.maximum(0, r - y)
    vy2 = np.maximum(0, r - (1 - y))
    
    bound_viol = vx1**2 + vx2**2 + vy1**2 + vy2**2
    obj += penalty_weight * np.sum(bound_viol)
    
    grad_x += penalty_weight * (-2.0 * vx1 + 2.0 * vx2)
    grad_y += penalty_weight * (-2.0 * vy1 + 2.0 * vy2)
    grad_r += penalty_weight * (2.0 * vx1 + 2.0 * vx2 + 2.0 * vy1 + 2.0 * vy2)
    
    grad = np.concatenate([grad_x, grad_y, grad_r])
    return obj, grad

def l_bfgsb_optimize(x, y, r, deadline, max_iter=2000):
    """
    Runs L-BFGS-B optimization starting from initial state x, y, r.
    Returns best score and config found during the run.
    """
    best_score = 0.0
    best_config = None
    
    z0 = np.concatenate([x, y, r])
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    penalty_weight = 100.0
    
    def callback(z):
        nonlocal best_score, best_config
        x_c = z[:N]
        y_c = z[N:2*N]
        r_c = z[2*N:]
        
        if is_valid(x_c, y_c, r_c):
            score = np.sum(r_c)
            if score > best_score:
                best_score = score
                best_config = (x_c.copy(), y_c.copy(), r_c.copy())
                
    try:
        # Check time before starting
        if time.time() > deadline - 5:
            return best_score, best_config
            
        res = minimize(
            objective_and_grad,
            z0,
            args=(penalty_weight,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        z_final = res.x
        x_f = z_final[:N]
        y_f = z_final[N:2*N]
        r_f = z_final[2*N:]
        
        if is_valid(x_f, y_f, r_f):
            score = np.sum(r_f)
            if score > best_score:
                best_score = score
                best_config = (x_f.copy(), y_f.copy(), r_f.copy())
                
    except Exception:
        pass
        
    return best_score, best_config

def simulated_annealing(x, y, r, start_time, deadline):
    """
    Simulated Annealing to escape local minima.
    Uses a temperature schedule and accepts worse solutions with probability exp(-delta_E/T).
    """
    best_score = 0.0
    best_config = None
    
    # Initial state
    cur_x = x.copy()
    cur_y = y.copy()
    cur_r = r.copy()
    
    # Initial score
    if is_valid(cur_x, cur_y, cur_r):
        best_score = np.sum(cur_r)
        best_config = (cur_x.copy(), cur_y.copy(), cur_r.copy())
    else:
        best_score = 0.0
        best_config = None
    
    # Temperature schedule
    T0 = 0.1
    Tend = 0.001
    n_steps = 5000
    cool_factor = (Tend / T0) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
            
        T = T0 * (cool_factor ** step)
        
        # Generate neighbor
        # Move a random subset of centers and adjust radii
        n_move = np.random.randint(1, 4)
        idxs = np.random.choice(N, n_move, replace=False)
        
        new_x = cur_x.copy()
        new_y = cur_y.copy()
        new_r = cur_r.copy()
        
        # Move centers
        for i in idxs:
            new_x[i] += np.random.normal(0, 0.05)
            new_y[i] += np.random.normal(0, 0.05)
            new_x[i] = np.clip(new_x[i], 0.01, 0.99)
            new_y[i] = np.clip(new_y[i], 0.01, 0.99)
            
        # Try to increase radii slightly using LP or simple projection
        # For SA, we need a valid state to compare energy.
        # Let's use LP to find max radii for these centers.
        new_r_lp = solve_radii_lp(new_x, new_y)
        
        if new_r_lp is not None:
            new_r = new_r_lp
            new_valid = True
        else:
            # Fallback: keep old radii or shrink them
            new_r = cur_r * 0.9
            new_valid = is_valid(new_x, new_y, new_r)
            if not new_valid:
                # Try to fix by shrinking more
                new_r = cur_r * 0.8
                new_valid = is_valid(new_x, new_y, new_r)
                if not new_valid:
                    continue # Skip invalid move
        
        cur_score = np.sum(new_r)
        
        # Acceptance criterion
        # We want to maximize sum(r), so "energy" is -sum(r)
        # delta_E = E_new - E_old = -cur_score - (-best_score) = best_score - cur_score
        # If cur_score > best_score, delta_E < 0, always accept.
        # If cur_score < best_score, delta_E > 0, accept with prob exp(-delta_E/T)
        
        # Note: We should compare with CURRENT state, not best state, for SA to work properly.
        # Let's track current score.
        cur_score_current = np.sum(cur_r) if is_valid(cur_x, cur_y, cur_r) else -1e9
        
        if is_valid(cur_x, cur_y, cur_r):
            delta_E = cur_score_current - cur_score # E_new - E_old (in terms of maximizing)
            # Wait, standard SA minimizes energy. Here we maximize score.
            # Let's define Energy = -Score.
            # E_old = -cur_score_current
            # E_new = -cur_score
            # delta_E = E_new - E_old = cur_score_current - cur_score
            # If delta_E < 0 (score increased), accept.
            # If delta_E > 0 (score decreased), accept with prob exp(-delta_E/T).
            
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / max(T, 1e-9)):
                cur_x = new_x
                cur_y = new_y
                cur_r = new_r
                if cur_score > best_score:
                    best_score = cur_score
                    best_config = (cur_x.copy(), cur_y.copy(), cur_r.copy())
        else:
            # Current state is invalid? Should not happen if we start valid and only accept valid.
            # But if we start invalid, we need to fix it.
            if cur_score > best_score:
                best_score = cur_score
                best_config = (cur_x.copy(), cur_y.copy(), cur_r.copy())
                
    return best_score, best_config

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = 0.0
    best_config = None
    
    # Strategy 1: L-BFGS-B from multiple initializations
    # Strategy 2: Simulated Annealing to escape local minima
    
    # Initializations
    inits = []
    
    # Grid
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            if len(xs) < N:
                xs.append((i + 0.5) / side)
                ys.append((j + 0.5) / side)
    inits.append((np.array(xs[:N]), np.array(ys[:N]), np.full(N, 0.02)))
    
    # Random
    inits.append((np.random.uniform(0.1, 0.9, N), np.random.uniform(0.1, 0.9, N), np.full(N, 0.02)))
    
    # Hexagonal
    rows = 5
    cols = 6
    xs = []
    ys = []
    for i in range(rows):
        for j in range(cols):
            if len(xs) < N:
                xs.append((j + 0.5 + (0.5 if i % 2 else 0)) / cols)
                ys.append((i + 0.5) / rows)
    inits.append((np.array(xs[:N]), np.array(ys[:N]), np.full(N, 0.02)))
    
    # Run L-BFGS-B on each init
    for x, y, r in inits:
        if time.time() > deadline - 20:
            break
        score, config = l_bfgsb_optimize(x, y, r, deadline)
        if config is not None:
            if score > best_score:
                best_score = score
                best_config = config
                
        # Refine with LP
        if config is not None:
            bx, by, br = config
            r_lp = solve_radii_lp(bx, by)
            if r_lp is not None:
                score_lp = np.sum(r_lp)
                if score_lp > best_score:
                    best_score = score_lp
                    best_config = (bx.copy(), by.copy(), r_lp.copy())
                    
    # Run Simulated Annealing from the best config found
    if best_config is not None and time.time() < deadline - 30:
        bx, by, br = best_config
        score_sa, config_sa = simulated_annealing(bx, by, br, start_time, deadline)
        if config_sa is not None:
            if score_sa > best_score:
                best_score = score_sa
                best_config = config_sa
                
        # Refine SA result with LP
        if config_sa is not None:
            bx, by, br = config_sa
            r_lp = solve_radii_lp(bx, by)
            if r_lp is not None:
                score_lp = np.sum(r_lp)
                if score_lp > best_score:
                    best_score = score_lp
                    best_config = (bx.copy(), by.copy(), r_lp.copy())
                    
    # Final fallback
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    bx, by, br = best_config
    return np.column_stack((bx, by, br))

# Import minimize
from scipy.optimize import minimize
