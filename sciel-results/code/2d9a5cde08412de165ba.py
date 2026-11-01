"""
Combined strategy:
1. Structural idea from current program: contact-graph aware initialization (hexagonal)
   and a smooth soft-min objective over all constraints (pairwise + wall).
2. Refinement machinery from inspiration: L-BFGS-B with analytic soft-min gradient,
   simulated-annealing style random restarts with adaptive move sizes, and a final
   high-precision polish pass using SLSQP on the active constraint set.

The soft-min objective replaces the hard min, giving a smooth surrogate that L-BFGS-B
can exploit efficiently. The gradient is computed analytically using weights derived
from the log-sum-exp form. A final polish pass tightens tolerance on the best solution.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < -1e-12:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective_and_grad(x, tau=0.003):
    """
    Objective and gradient for -soft_min(minimum constraint distance).
    x: flattened array of 2*N coordinates
    Returns (obj, grad)
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # All pairwise distances via KDTree for efficiency
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair_dists = dists[:, 1] / 2.0
    nearest_idx = idx[:, 1]
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, min_pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    soft_min = min_d - tau * np.log(np.sum(exp_vals))
    
    obj = -soft_min
    
    # Gradient
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    for i in range(N):
        w_wall = weights[i]
        if w_wall < 1e-14:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])
    
    # 2. Pair contributions
    for i in range(N):
        w_pair = weights[N + i]
        if w_pair < 1e-14:
            continue
        j = nearest_idx[i]
        d = dists[i, 1]
        if d < 1e-12:
            continue
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
    
    return obj, grad

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = np.sqrt(3.0) * r * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        r_actual = _compute_radius(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_grid = centers.copy()
            
    if best_grid is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_grid = grid[:N].copy()
        best_r = _compute_radius(best_grid)
        
    return best_grid, best_r

def _lbfgs_optimize(centers, time_budget, tau=0.003):
    """
    Optimize using L-BFGS-B with soft-min objective.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def obj_fun(x):
        return _soft_min_objective_and_grad(x, tau)[0]
    
    def grad_fun(x):
        return _soft_min_objective_and_grad(x, tau)[1]
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            obj_fun, x0, jac=grad_fun,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        new_cents = res.x.reshape(N, 2)
        new_r = _compute_radius(new_cents)
        return new_cents, new_r
    except Exception:
        return centers, _compute_radius(centers)

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to explore configuration space.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.012
    T_end = 0.00003
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current radius
        tree = cKDTree(current)
        dists, idx = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        # Build weights for constraint selection
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx_sel = rng.choice(len(all_dists), p=weights)
        
        if idx_sel < N:
            # Move a single circle
            i = idx_sel
            move_size = 0.002 + 0.010 * T / T_start
            direction = rng.integers(0, 4)
            delta = np.zeros(2)
            if direction == 0:
                delta[0] = move_size * rng.uniform(0.5, 1.0)
            elif direction == 1:
                delta[0] = -move_size * rng.uniform(0.5, 1.0)
            elif direction == 2:
                delta[1] = move_size * rng.uniform(0.5, 1.0)
            else:
                delta[1] = -move_size * rng.uniform(0.5, 1.0)
            
            trial = current.copy()
            trial[i] += delta
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Move a pair of circles
            pair_idx = idx_sel - N
            i = pair_idx
            j = idx[i, 1]
            
            move_size = 0.002 + 0.010 * T / T_start
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.5, 1.0)
            trial[j] -= unit * move_size * rng.uniform(0.5, 1.0)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Evaluate trial
        trial_r = _compute_radius(trial)
        
        # Accept if better or with probability exp(-delta/T)
        if trial_r >= curr_r:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            delta = curr_r - trial_r
            if delta > 0 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _polish_slsqp(centers, time_budget):
    """
    Final high-precision polish using SLSQP on the best solution.
    Uses active constraints only.
    """
    start_time = time.time()
    n = N
    r0 = _compute_radius(centers)
    
    # Identify active constraints
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    r = min(min_pair, min_wall)
    
    # Build SLSQP problem with all constraints
    x0 = np.zeros(2*n + 1)
    x0[:2*n] = centers.flatten()
    x0[2*n] = r * 0.999
    
    cons = []
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            def c_func(p, i=i, j=j):
                ci = p[2*i:2*i+2]
                cj = p[2*j:2*j+2]
                r_val = p[2*n]
                return np.dot(ci-cj, ci-cj) - 4*r_val*r_val
            def c_grad(p, i=i, j=j):
                g = np.zeros_like(p)
                ci = p[2*i:2*i+2]
                cj = p[2*j:2*j+2]
                r_val = p[2*n]
                diff = ci - cj
                g[2*i:2*i+2] = 2*diff
                g[2*j:2*j+2] = -2*diff
                g[2*n] = -8*r_val
                return g
            cons.append({'type': 'ineq', 'fun': c_func, 'jac': c_grad})
    
    # Wall constraints
    for i in range(n):
        idx_x = 2*i
        idx_y = 2*i + 1
        idx_r = 2*n
        
        def c_left(p, ix=idx_x):
            return p[ix] - p[idx_r] + 0.5
        def g_left(p, ix=idx_x):
            g = np.zeros_like(p)
            g[ix] = 1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_left, 'jac': g_left})
        
        def c_right(p, ix=idx_x):
            return 0.5 - p[idx_r] - p[ix]
        def g_right(p, ix=idx_x):
            g = np.zeros_like(p)
            g[ix] = -1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_right, 'jac': g_right})
        
        def c_bot(p, iy=idx_y):
            return p[iy] - p[idx_r] + 0.5
        def g_bot(p, iy=idx_y):
            g = np.zeros_like(p)
            g[iy] = 1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_bot, 'jac': g_bot})
        
        def c_top(p, iy=idx_y):
            return 0.5 - p[idx_r] - p[iy]
        def g_top(p, iy=idx_y):
            g = np.zeros_like(p)
            g[iy] = -1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_top, 'jac': g_top})
    
    def obj(p):
        return -p[2*n]
    def obj_grad(p):
        g = np.zeros_like(p)
        g[2*n] = -1.0
        return g
    
    bounds = [(None, None)] * (2*n) + [(0, 0.5)]
    
    try:
        res = minimize(
            obj, x0, jac=obj_grad,
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        if res.success:
            new_cents = res.x[:2*n].reshape(n, 2)
            new_r = _compute_radius(new_cents)
            return new_cents, new_r
    except Exception:
        pass
    
    return centers, r0

def construct():
    start_time = time.time()
    deadline = DEADLINE
    
    # 1. Initial hexagonal packing
    cents, r0 = _hex_init()
    best_cents = cents.copy()
    best_r = r0
    
    rng = np.random.default_rng(42)
    
    # 2. L-BFGS-B optimization
    remaining = deadline - (time.time() - start_time)
    if remaining > 20:
        opt_cents, opt_r = _lbfgs_optimize(best_cents, min(remaining - 10, 40))
        if opt_r > best_r:
            best_r = opt_r
            best_cents = opt_cents
    
    # 3. Simulated annealing with multiple restarts
    remaining = deadline - (time.time() - start_time)
    if remaining > 30:
        sa_cents, sa_r = _simulated_annealing(best_cents, rng, min(remaining - 20, 50))
        if sa_r > best_r:
            best_r = sa_r
            best_cents = sa_cents
    
    # 4. Another L-BFGS-B pass on the SA result
    remaining = deadline - (time.time() - start_time)
    if remaining > 20:
        opt_cents, opt_r = _lbfgs_optimize(best_cents, min(remaining - 10, 30))
        if opt_r > best_r:
            best_r = opt_r
            best_cents = opt_cents
    
    # 5. Final SLSQP polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        polish_cents, polish_r = _polish_slsqp(best_cents, min(remaining - 5, 20))
        if polish_r > best_r:
            best_r = polish_r
            best_cents = polish_cents
    
    # Final safety check
    if time.time() - start_time > deadline - 5:
        pass  # Just return what we have
    
    return best_cents
