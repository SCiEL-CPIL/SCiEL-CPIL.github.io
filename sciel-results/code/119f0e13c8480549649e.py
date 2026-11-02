"""Packing 200 equal circles in a unit square using hybrid soft-min + SLSQP constrained optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.02):
    """Smooth energy function using log-sum-exp soft minimum."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min over pairwise distances using a window around the minimum
    d_min_all = np.min(dists)
    window = 0.04
    mask = dists < (d_min_all + window)
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # Soft min over boundary distances
    b_min = np.min(boundary_dists)
    b_window = 0.04
    b_mask = boundary_dists < (b_min + b_window)
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = min_boundary
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of the two radii
    r_min = min(r_pair, r_bound)
    arr = np.array([r_pair, r_bound])
    soft_min_combined = r_min - T * np.log(np.sum(np.exp(-(arr - r_min) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, searching over multiple spacings."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _jittered_grid_init(seed=0):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng.shuffle(grid)
    c = grid[:N].copy()
    c += rng.normal(0, 0.01, size=(N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    return c


def _lbfgs_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with soft-min energy, multi-temperature schedule."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    temperatures = [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003]
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 3:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 150}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _pdist_to_ij(k, n):
    """Convert pdist index to (i, j) pair index."""
    # pdist uses upper triangular: index k corresponds to (i, j) where
    # k = i*(2*n-1-i)//2 + (j - i - 1)
    # Solve for i: i*(2n-1-i)//2 <= k
    # Approximate: i ≈ n - 0.5 - sqrt((n-0.5)^2 - 2k)
    i = int(n - 1 - np.sqrt(8 * k + 1)) // 2
    # Verify and adjust
    while i < n - 1:
        base = i * (2 * n - 1 - i) // 2
        if base + (n - 1 - i) > k:
            break
        i += 1
    j_offset = k - i * (2 * n - 1 - i) // 2
    j = i + 1 + j_offset
    return i, j


def _slsqp_polish(c0, max_time=30.0):
    """
    SLSQP-based polish: maximize radius t subject to pairwise and boundary constraints.
    Uses active-set identification to keep the constraint set manageable.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    c_current = c0.copy()
    r_current = best_r
    
    max_rounds = 3
    
    for round_idx in range(max_rounds):
        if time.time() > start_time + max_time - 5:
            break
        
        # Identify active constraints
        t_ref = r_current
        dists = pdist(c_current, metric='euclidean')
        
        # Close pairs: within 2.2 * t_ref
        close_pairs_mask = dists < 2.2 * t_ref
        close_pairs_idx = np.where(close_pairs_mask)[0]
        
        # Limit to at most 300 closest pairs
        if len(close_pairs_idx) > 300:
            # Get the 300 closest
            sub_dists = dists[close_pairs_idx]
            order = np.argsort(sub_dists)[:300]
            close_pairs_idx = close_pairs_idx[order]
        
        # Boundary constraints: circles close to boundary
        boundary_dists = 0.5 - np.abs(c_current)  # shape (N, 2)
        flat_bd = boundary_dists.ravel()
        close_bound_mask = flat_bd < 1.5 * t_ref
        close_bound_idx = np.where(close_bound_mask)[0]
        
        # Limit boundary constraints
        if len(close_bound_idx) > 100:
            sub_bd = flat_bd[close_bound_idx]
            order = np.argsort(sub_bd)[:100]
            close_bound_idx = close_bound_idx[order]
        
        # Build the SLSQP problem: maximize t
        # Variables: [t, c_0x, c_0y, c_1x, c_1y, ...]
        n_vars = 1 + 2 * N
        
        def objective(x):
            # Minimize -t
            return -x[0]
        
        constraints = []
        
        # Pairwise constraints: ||c_i - c_j|| >= 2t
        for k in close_pairs_idx:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros(n_vars)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        # Boundary constraints: 0.5 - |c_ik| >= t
        for k in close_bound_idx:
            i = k // 2
            axis = k % 2
            
            def _bound_fun(x, _i=i, _k=axis):
                t = x[0]
                c_val = x[1 + _i*2 + _k]
                return 0.5 - np.abs(c_val) - t
            
            def _bound_jac(x, _i=i, _k=axis):
                grad = np.zeros(n_vars)
                grad[0] = -1.0
                c_val = x[1 + _i*2 + _k]
                if c_val >= 0:
                    grad[1 + _i*2 + _k] = -1.0
                else:
                    grad[1 + _i*2 + _k] = 1.0
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        if not constraints:
            break
        
        bounds = [(0.0, 0.5)] + [(-0.499999, 0.499999)] * (2 * N)
        
        x0 = np.zeros(n_vars)
        x0[0] = r_current
        x0[1:] = c_current.ravel()
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.status == 0:
                t_new = result.x[0]
                c_new = result.x[1:].reshape(N, 2)
                c_new = np.clip(c_new, -0.499999, 0.499999)
                
                r_actual = _compute_radius(c_new)
                
                if r_actual > best_r:
                    best_r = r_actual
                    best_c = c_new.copy()
                
                c_current = c_new.copy()
                r_current = r_actual
            else:
                # Try with fewer constraints
                if len(constraints) > 20:
                    # Keep only the most violated
                    continue
                break
        except Exception:
            break
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    for T in [0.004, 0.002, 0.001]:
        if time.time() > start_time + max_time - 3:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _local_search_refine(c0, max_time=15.0):
    """
    Local search: iteratively find the most-violated constraint and move the
    involved circles to relieve it. This is a constraint-directed local search.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    c_current = c0.copy()
    r_current = best_r
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > start_time + max_time - 2:
            break
        
        # Compute current radius and identify the binding constraint
        dists = pdist(c_current, metric='euclidean')
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        
        boundary_dists = 0.5 - np.abs(c_current)
        min_bound_idx = np.argmin(boundary_dists)
        min_bound_val = boundary_dists.ravel()[min_bound_idx]
        
        r_pair = min_pair_dist / 2.0
        r_bound = min_bound_val
        r_current = min(r_pair, r_bound)
        
        if r_current > best_r:
            best_r = r_current
            best_c = c_current.copy()
        
        # Determine which constraint is binding
        if r_pair <= r_bound:
            # Pairwise constraint is binding
            i, j = _pdist_to_ij(min_pair_idx, N)
            diff = c_current[i] - c_current[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                direction = np.array([1.0, 0.0])
            else:
                direction = diff / d
            
            # Try moving i and j apart
            step = 0.001
            trial = c_current.copy()
            trial[i] += step * direction
            trial[j] -= step * direction
            trial = np.clip(trial, -0.499999, 0.499999)
            
            r_trial = _compute_radius(trial)
            if r_trial > r_current:
                c_current = trial
            else:
                # Try smaller step
                step *= 0.5
                trial = c_current.copy()
                trial[i] += step * direction
                trial[j] -= step * direction
                trial = np.clip(trial, -0.499999, 0.499999)
                r_trial = _compute_radius(trial)
                if r_trial > r_current:
                    c_current = trial
                else:
                    # Try moving just one
                    trial = c_current.copy()
                    trial[i] += step * 0.5 * direction
                    trial = np.clip(trial, -0.499999, 0.499999)
                    r_trial = _compute_radius(trial)
                    if r_trial > r_current:
                        c_current = trial
                    else:
                        trial = c_current.copy()
                        trial[j] -= step * 0.5 * direction
                        trial = np.clip(trial, -0.499999, 0.499999)
                        r_trial = _compute_radius(trial)
                        if r_trial > r_current:
                            c_current = trial
                        else:
                            break  # Can't improve
        else:
            # Boundary constraint is binding
            i = min_bound_idx // 2
            axis = min_bound_idx % 2
            c_val = c_current[i, axis]
            
            # Push away from boundary
            if c_val >= 0:
                direction = -1.0
            else:
                direction = 1.0
            
            step = 0.001
            trial = c_current.copy()
            trial[i, axis] += step * direction
            trial = np.clip(trial, -0.499999, 0.499999)
            
            r_trial = _compute_radius(trial)
            if r_trial > r_current:
                c_current = trial
            else:
                # Try smaller step
                step *= 0.5
                trial = c_current.copy()
                trial[i, axis] += step * direction
                trial = np.clip(trial, -0.499999, 0.499999)
                r_trial = _compute_radius(trial)
                if r_trial > r_current:
                    c_current = trial
                else:
                    break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: L-BFGS optimization from hex
    if time.time() < deadline - 60:
        remaining = deadline - time.time() - 30
        max_opt_time = min(25.0, remaining)
        c_opt, r_opt = _lbfgs_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: SLSQP constrained polish
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 20
        if remaining > 10:
            c_sls, r_sls = _slsqp_polish(best_c, max_time=min(30.0, remaining))
            if r_sls > best_radius:
                best_radius = r_sls
                best_c = c_sls.copy()
    
    # Step 4: Local search refinement
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 15
        if remaining > 8:
            c_ls, r_ls = _local_search_refine(best_c, max_time=min(15.0, remaining))
            if r_ls > best_radius:
                best_radius = r_ls
                best_c = c_ls.copy()
    
    # Step 5: Second SLSQP round
    if time.time() < deadline - 25:
        remaining = deadline - time.time() - 12
        if remaining > 8:
            c_sls2, r_sls2 = _slsqp_polish(best_c, max_time=min(20.0, remaining))
            if r_sls2 > best_radius:
                best_radius = r_sls2
                best_c = c_sls2.copy()
    
    # Step 6: Random restarts with L-BFGS
    n_restarts = 2
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        c_init = _jittered_grid_init(seed=42 + trial)
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 15
        if remaining > 8:
            max_opt_time = min(15.0, remaining)
            c_opt, r_opt = _lbfgs_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 7: Final SLSQP polish
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 10
        if remaining > 8:
            c_sls3, r_sls3 = _slsqp_polish(best_c, max_time=min(15.0, remaining))
            if r_sls3 > best_radius:
                best_radius = r_sls3
                best_c = c_sls3.copy()
    
    # Step 8: Final local search
    if time.time() < deadline - 12:
        remaining = deadline - time.time() - 6
        if remaining > 5:
            c_ls2, r_ls2 = _local_search_refine(best_c, max_time=min(10.0, remaining))
            if r_ls2 > best_radius:
                best_radius = r_ls2
                best_c = c_ls2.copy()
    
    # Step 9: Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 4:
            c_pol, r_pol = _polish(best_c, max_time=min(8.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
