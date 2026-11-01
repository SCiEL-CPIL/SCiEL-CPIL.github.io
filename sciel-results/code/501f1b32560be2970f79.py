"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using smooth soft-min surrogate.
3. Focused local optimization on near-active constraints.
4. Simulated annealing with rigid-body cluster moves.
5. Ultra-precision final polish.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """Compute soft-min of values array using log-sum-exp."""
    v = np.asarray(values)
    vmin = np.min(v)
    w = np.exp(-beta * (v - vmin))
    w /= np.sum(w)
    return vmin + np.log(np.sum(w)) / beta, w

def _smooth_objective(params, beta=200.0):
    """
    Smooth objective: minimize soft-min of (pair distances / 2, boundary distances).
    This approximates -max_radius.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Boundary distances
    bdists = np.minimum(
        np.minimum(0.5 - x, 0.5 + x),
        np.minimum(0.5 - y, 0.5 + y)
    )
    
    # Combine all constraint values: pair distances (as radius = d/2) and boundary distances
    # We want to maximize min(d_pairs/2, bdists)
    # So minimize -soft_min(d_pairs/2, bdists)
    # Equivalently: soft_min of negative values, or use log-sum-exp on negatives
    
    # Use: E = soft_min_neg = -soft_min of the values
    # soft_min_neg(v) = -(1/beta)*log(sum(exp(-beta*v))) ... wait no
    # soft_min(v) = (1/beta)*log(sum(exp(-beta*v))) - this is small when all v are large
    # We want to maximize min, so minimize -min
    # -soft_min(v) = -(1/beta)*log(sum(exp(-beta*v)))
    
    # Actually simpler: just compute soft_min of all constraint values
    # and minimize its negative... no, minimize soft_min directly since it's a lower bound on min
    
    # Let's just minimize soft_min of (d_pairs/2) and soft_min of bdists separately
    # and sum them (or take the min of the two soft-mins)
    
    # Better: combine all into one soft-min
    all_vals = np.concatenate([d_pairs / 2.0, bdists])
    
    vmin = np.min(all_vals)
    w = np.exp(-beta * (all_vals - vmin))
    w_sum = np.sum(w)
    w /= w_sum
    
    E = vmin + np.log(w_sum) / beta
    
    # Gradient
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    # Pair contribution
    p_pairs = w[:len(d_pairs)]
    grad_x[i_idx] -= p_pairs * (vec_x * inv_d) / 2.0
    grad_x[j_idx] += p_pairs * (vec_x * inv_d) / 2.0
    grad_y[i_idx] -= p_pairs * (vec_y * inv_d) / 2.0
    grad_y[j_idx] += p_pairs * (vec_y * inv_d) / 2.0
    
    # Boundary contribution
    p_bounds = w[len(d_pairs):]
    
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
    d_min_b = np.min(d_all, axis=1, keepdims=True)
    wall_w = np.exp(-beta * (d_all - d_min_b))
    wall_w /= wall_w.sum(axis=1, keepdims=True)
    
    # For each point, the boundary gradient depends on which wall is closest
    # But we already have the combined weight, so we need to decompose
    # Actually, let's just compute the gradient of soft_min(bdists) directly
    
    # Recompute boundary soft-min gradient properly
    bd_shifted = bdists - np.min(bdists)
    w_b = np.exp(-beta * bd_shifted)
    w_b_sum = np.sum(w_b)
    p_b = w_b / w_b_sum
    
    # d(bdists[i])/dx[i] = -1 if closest wall is left, +1 if right
    # d(bdists[i])/dy[i] = -1 if closest wall is down, +1 if up
    
    # For each point, determine which wall is active
    min_wall_idx = np.argmin(d_all, axis=1)  # 0=left, 1=right, 2=down, 3=up
    
    for i in range(N):
        if p_b[i] < 1e-15:
            continue
        mw = min_wall_idx[i]
        if mw == 0:
            grad_x[i] -= p_b[i]
        elif mw == 1:
            grad_x[i] += p_b[i]
        elif mw == 2:
            grad_y[i] -= p_b[i]
        else:
            grad_y[i] += p_b[i]
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E, total_grad

def _energy_focused(params, beta=300.0, focus_threshold=1.5):
    """
    Focused energy: only consider constraints near the minimum.
    This is much faster than full pairwise computation.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Compute current radius estimate
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    bdists = np.minimum(
        np.minimum(0.5 - x, 0.5 + x),
        np.minimum(0.5 - y, 0.5 + y)
    )
    
    r_est = min(np.min(d_pairs) / 2.0, np.min(bdists))
    
    # Focus on pairs with distance < 2 * r_est * focus_threshold
    focus_mask_pairs = d_pairs < 2.0 * r_est * focus_threshold
    focus_pairs_idx = np.where(focus_mask_pairs)[0]
    
    # Focus on boundaries with distance < r_est * focus_threshold
    focus_mask_bounds = bdists < r_est * focus_threshold
    focus_bounds_idx = np.where(focus_mask_bounds)[0]
    
    if len(focus_pairs_idx) == 0 and len(focus_bounds_idx) == 0:
        # No focused constraints, use full objective
        return _smooth_objective(params, beta=beta)
    
    # Build focused distance array
    d_focused = d_pairs[focus_pairs_idx]
    
    # Boundary focused
    b_focused = bdists[focus_bounds_idx]
    
    all_vals = np.concatenate([d_focused / 2.0, b_focused])
    
    vmin = np.min(all_vals)
    w = np.exp(-beta * (all_vals - vmin))
    w_sum = np.sum(w)
    w /= w_sum
    
    E = vmin + np.log(w_sum) / beta
    
    # Gradient
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    if len(focus_pairs_idx) > 0:
        i_idx = iu[0][focus_pairs_idx]
        j_idx = iu[1][focus_pairs_idx]
        
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        inv_d = 1.0 / np.maximum(d_focused, 1e-15)
        
        p_pairs = w[:len(d_focused)]
        grad_x[i_idx] -= p_pairs * (vec_x * inv_d) / 2.0
        grad_x[j_idx] += p_pairs * (vec_x * inv_d) / 2.0
        grad_y[i_idx] -= p_pairs * (vec_y * inv_d) / 2.0
        grad_y[j_idx] += p_pairs * (vec_y * inv_d) / 2.0
    
    if len(focus_bounds_idx) > 0:
        p_bounds = w[len(d_focused):]
        
        d_left = 0.5 - x[focus_bounds_idx]
        d_right = 0.5 + x[focus_bounds_idx]
        d_down = 0.5 - y[focus_bounds_idx]
        d_up = 0.5 + y[focus_bounds_idx]
        
        d_all_f = np.stack([d_left, d_right, d_down, d_up], axis=1)
        min_wall_idx = np.argmin(d_all_f, axis=1)
        
        for idx, i in enumerate(focus_bounds_idx):
            if p_bounds[idx] < 1e-15:
                continue
            mw = min_wall_idx[idx]
            if mw == 0:
                grad_x[i] -= p_bounds[idx]
            elif mw == 1:
                grad_x[i] += p_bounds[idx]
            elif mw == 2:
                grad_y[i] -= p_bounds[idx]
            else:
                grad_y[i] += p_bounds[idx]
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0
    
    # Try different densities with finer granularity around optimal region
    for d in np.linspace(0.052, 0.075, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, time_budget=20.0, beta=100.0):
    """Run L-BFGS-B with smooth objective."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        def fun_and_grad(params):
            return _smooth_objective(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 3000, 'maxls': 50}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

def _local_optimize_smooth(c, beta=200.0, max_iter=200):
    """Local optimization using smooth objective with L-BFGS-B."""
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        res = minimize(
            lambda p: _smooth_objective(p, beta=beta),
            c.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 2000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c

def _local_optimize_focused(c, beta=300.0, focus_threshold=1.5, max_iter=300):
    """Local optimization using focused energy (only near-active constraints)."""
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        res = minimize(
            lambda p: _energy_focused(p, beta=beta, focus_threshold=focus_threshold),
            c.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 3000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c

def _get_active_pairs(c, r, tol=1e-7):
    """Get pairs of circles that are close to touching."""
    dists = pdist(c, metric='euclidean')
    dist_mat = squareform(dists)
    np.fill_diagonal(dist_mat, np.inf)
    
    active = dist_mat < (2 * r + tol)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if active[i, j]:
                pairs.append((i, j))
    return pairs

def _slsqp_polish(c, r_target, time_limit=10.0):
    """
    SLSQP polish: try to satisfy all constraints for radius r_target.
    """
    start_time = time.time()
    c = c.copy()
    
    active_pairs = _get_active_pairs(c, r_target)
    
    if len(active_pairs) > 100:
        # Too many active constraints, skip
        return c, False
    
    cons = []
    for i, j in active_pairs:
        def constraint_pair(x, i=i, j=j):
            xi = x[2*i:2*i+2]
            xj = x[2*j:2*j+2]
            return np.dot(xi-xj, xi-xj) - (2*r_target)**2
        cons.append({'type': 'ineq', 'fun': constraint_pair})
    
    # Boundary constraints
    bdists = np.minimum(
        np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
        np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
    )
    active_bounds = np.where(bdists < (r_target + 1e-7))[0]
    
    for i in active_bounds:
        d_left = 0.5 - c[i, 0]
        d_right = c[i, 0] + 0.5
        d_down = 0.5 - c[i, 1]
        d_up = c[i, 1] + 0.5
        min_d = min(d_left, d_right, d_down, d_up)
        
        if d_left <= d_right and d_left <= d_down and d_left <= d_up:
            def constraint_wall(x, i=i, r=r_target):
                return 0.5 - x[2*i] - r
            cons.append({'type': 'ineq', 'fun': constraint_wall})
        elif d_right <= d_left and d_right <= d_down and d_right <= d_up:
            def constraint_wall(x, i=i, r=r_target):
                return x[2*i] + 0.5 - r
            cons.append({'type': 'ineq', 'fun': constraint_wall})
        elif d_down <= d_left and d_down <= d_right and d_down <= d_up:
            def constraint_wall(x, i=i, r=r_target):
                return 0.5 - x[2*i+1] - r
            cons.append({'type': 'ineq', 'fun': constraint_wall})
        else:
            def constraint_wall(x, i=i, r=r_target):
                return x[2*i+1] + 0.5 - r
            cons.append({'type': 'ineq', 'fun': constraint_wall})
    
    bounds = [(-0.5 + r_target, 0.5 - r_target)] * (2 * N)
    
    def objective(x):
        return 0.0
    
    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-14}
        )
        
        if res.success:
            c_new = res.x.reshape(N, 2)
            r_new = _compute_radius(c_new)
            if r_new >= r_target - 1e-10:
                return c_new, True
    except Exception:
        pass
        
    return c, False

def _simulated_annealing_rigid(c, start_time, time_limit, rng, T_start=0.0005, T_end=0.00001):
    """
    Simulated annealing with rigid-body cluster moves.
    """
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    n_steps = 50
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Temperature schedule
        progress = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random cluster move
        c_new = c.copy()
        
        # Pick a random cluster of nearby circles
        i = rng.randint(N)
        # Find neighbors within 3*r
        dx = c_new[:, 0] - c_new[i, 0]
        dy = c_new[:, 1] - c_new[i, 1]
        dist = np.sqrt(dx**2 + dy**2)
        neighbors = np.where(dist < 3 * current_r)[0]
        neighbors = neighbors[neighbors != i]
        
        if len(neighbors) > 0:
            # Move the cluster
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = rng.uniform(0.002, 0.01)
            dx_move = magnitude * np.cos(angle)
            dy_move = magnitude * np.sin(angle)
            
            for j in [i] + list(neighbors):
                c_new[j, 0] += dx_move
                c_new[j, 1] += dy_move
        
        # Clip
        c_new = np.clip(c_new, -0.49999, 0.49999)
        r_new = _compute_radius(c_new)
        
        if r_new > current_r:
            c = c_new
            current_r = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            delta = current_r - r_new
            if delta < T and rng.random() < np.exp(-delta / max(T, 1e-12)):
                c = c_new
                current_r = r_new
        
        # Periodically run local optimization
        if step % 5 == 4 and time.time() - start_time < time_limit - 5:
            c_opt = _local_optimize_focused(c, beta=300.0, focus_threshold=1.5, max_iter=100)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c.copy()
    
    return best_c, best_r

def _ultra_polish(c, time_limit=10.0):
    """
    Ultra-precision polish using very high beta and tight tolerances.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    # Run with increasing beta
    for beta in [500.0, 1000.0, 2000.0]:
        if time.time() - start_time > time_limit - 2:
            break
        
        c_opt = _local_optimize_smooth(c, beta=beta, max_iter=300)
        r_opt = _compute_radius(c_opt)
        if r_opt > current_r:
            c = c_opt
            current_r = r_opt
    
    # SLSQP polish attempts
    if time.time() - start_time < time_limit - 1:
        for step in [1e-7, 5e-8, 1e-8]:
            if time.time() - start_time > time_limit - 1:
                break
            target_r = current_r + step
            c_new, success = _slsqp_polish(c, target_r, time_limit=5.0)
            if success:
                c = c_new
                current_r = _compute_radius(c)
            else:
                break
    
    return c, current_r

def _randomize_clusters(c, rng, n_clusters=3, magnitude_range=(0.003, 0.01)):
    """Randomize positions of a few clusters to escape local minima."""
    c_new = c.copy()
    
    for _ in range(n_clusters):
        i = rng.randint(N)
        dx = c_new[:, 0] - c_new[i, 0]
        dy = c_new[:, 1] - c_new[i, 1]
        dist = np.sqrt(dx**2 + dy**2)
        neighbors = np.where(dist < 0.1)[0]
        neighbors = neighbors[neighbors != i]
        
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(*magnitude_range)
        dx_move = magnitude * np.cos(angle)
        dy_move = magnitude * np.sin(angle)
        
        for j in [i] + list(neighbors):
            c_new[j, 0] += dx_move
            c_new[j, 1] += dy_move
    
    return np.clip(c_new, -0.49999, 0.49999)

def _local_polish(c, time_limit=30.0):
    """
    Local polish combining focused optimization, simulated annealing, and ultra-polish.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    rng = np.random.RandomState(42)
    
    # Phase 1: Focused local optimization
    if time.time() - start_time < time_limit * 0.4:
        for beta in [200.0, 300.0, 500.0]:
            if time.time() - start_time > time_limit * 0.4:
                break
            c_opt = _local_optimize_focused(c, beta=beta, focus_threshold=1.5, max_iter=200)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c.copy()
    
    # Phase 2: Simulated annealing with rigid moves
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        c_annealed, r_annealed = _simulated_annealing_rigid(
            c, start_time, time.time() + remaining * 0.5, rng
        )
        if r_annealed > best_r:
            best_r = r_annealed
            best_c = c_annealed.copy()
            c = c_annealed
            current_r = r_annealed
    
    # Phase 3: Ultra-precision polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        c_ultra, r_ultra = _ultra_polish(c, time_limit=min(remaining, 15.0))
        if r_ultra > best_r:
            best_r = r_ultra
            best_c = c_ultra.copy()
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Phase 1: Global optimization with multiple beta values
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 100:
            break
        c_opt = _optimize_energy(best_c, time_budget=15.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Phase 2: Perturbed restarts
    rng = np.random.RandomState(42)
    remaining = deadline - time.time() - 60
    if remaining > 20:
        for attempt in range(3):
            if time.time() > deadline - 60:
                break
            c_perturbed = _randomize_clusters(best_c, rng, n_clusters=2, magnitude_range=(0.005, 0.012))
            c_opt2 = _optimize_energy(c_perturbed, time_budget=10.0, beta=150.0)
            r_opt2 = _compute_radius(c_opt2)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()
    
    # Phase 3: Local polish (main)
    remaining = deadline - time.time() - 5
    if remaining > 20:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 70.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Phase 4: Second polish pass
    remaining = deadline - time.time() - 3
    if remaining > 15:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 30.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Phase 5: Final ultra-polish
    remaining = deadline - time.time() - 2
    if remaining > 5:
        c_ultra, r_ultra = _ultra_polish(best_c, time_limit=min(remaining, 15.0))
        if r_ultra > best_r:
            best_r = r_ultra
            best_c = c_ultra.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
