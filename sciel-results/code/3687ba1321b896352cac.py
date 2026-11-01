"""50-sphere packing in a unit cube using smooth surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _neg_radius(c):
    """Negative radius for minimization."""
    return -_compute_radius(c)

def _soft_min_radius(c, beta):
    """
    Compute a smooth surrogate for the radius using soft-min.
    r_soft = -1/beta * log( sum_i exp(-beta * d_i/2) + sum_j exp(-beta * w_j) )
    where d_i are pairwise distances and w_j are wall distances.
    As beta -> inf, this approaches the true min.
    """
    c = np.asarray(c).reshape(-1, 3)
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    
    # Soft-min of pairwise distances / 2
    # We want min over i<j of dists[i,j]/2
    # Use: -1/beta * log(sum exp(-beta * dists[i,j]/2))
    # But we need to be careful with the sum over all pairs
    
    # For efficiency, use a two-stage soft-min
    # First, for each sphere i, compute soft-min distance to others
    # Then soft-min over all spheres
    
    # Actually, let's just do it directly but with numerical care
    # dists[i,j] for i != j
    d_half = dists_sq / 4.0  # (dists/2)^2... no, dists^2/4 = (dists/2)^2
    # We want exp(-beta * dists/2) = exp(-beta * sqrt(dists_sq)/2)
    # This is expensive. Let's use a different approach.
    
    # Alternative: use the log-sum-exp trick on the distances themselves
    # But sqrt is needed. Let's compute distances.
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of all pairwise distances / 2
    # = -1/beta * log( sum_{i<j} exp(-beta * dists[i,j]/2) )
    # Note: each pair appears twice in the full matrix, so we divide by 2 in the sum
    # sum_{i!=j} = 2 * sum_{i<j}
    # So log(sum_{i!=j} ...) = log(2) + log(sum_{i<j} ...)
    # The log(2) term is constant and doesn't affect optimization much,
    # but let's be precise.
    
    # For wall distances
    w = wall_dists.flatten()
    
    # Combine
    # r_soft = -1/beta * log( sum_{i<j} exp(-beta*d_ij/2) + sum_i exp(-beta*w_i) )
    # Note: the wall sum has N terms, pair sum has N*(N-1)/2 terms
    
    # Numerical stability: shift by max
    d_vals = dists[np.triu_indices(n, k=1)] / 2.0
    w_vals = w
    
    all_vals = np.concatenate([d_vals, w_vals])
    max_val = np.max(all_vals)
    if max_val < 1e-15:
        return 0.0
    
    exp_terms = np.exp(-beta * (all_vals - max_val))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-300:
        return max_val
    
    r_soft = max_val - np.log(sum_exp) / beta
    return r_soft

def _soft_min_radius_grad(c, beta):
    """
    Gradient of the soft-min radius surrogate.
    Returns (r_soft, grad) where grad has shape (N, 3).
    """
    c = np.asarray(c).reshape(-1, 3)
    n = len(c)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = 0.5 - np.abs(c)
    
    d_vals = dists[np.triu_indices(n, k=1)] / 2.0
    w_vals = wall_dists.flatten()
    
    all_vals = np.concatenate([d_vals, w_vals])
    max_val = np.max(all_vals)
    
    exp_terms = np.exp(-beta * (all_vals - max_val))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-300:
        return max_val, np.zeros_like(c)
    
    r_soft = max_val - np.log(sum_exp) / beta
    
    # Gradient computation
    # d/dx [ -1/beta * log(sum exp(-beta * f_i(x))) ]
    # = -1/beta * (1/sum_exp) * sum_j (-beta * exp(-beta*f_j) * df_j/dx)
    # = sum_j exp(-beta*f_j) * df_j/dx / sum_exp
    # = sum_j p_j * df_j/dx where p_j = exp(-beta*f_j)/sum_exp
    
    # For pairwise distances: f_ij = ||c_i - c_j||/2
    # df_ij/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # df_ij/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # For wall distances: f_i = 0.5 - |c_i_k| for the min over k
    # This is tricky because it's a min over coordinates.
    # Actually, wall_dists[i,k] = 0.5 - |c[i,k]|
    # The soft-min is over all (i,k) pairs.
    
    # Let's compute weights for each term
    n_pairs = len(d_vals)
    n_walls = len(w_vals)
    
    p_d = exp_terms[:n_pairs] / sum_exp
    p_w = exp_terms[n_pairs:] / sum_exp
    
    grad = np.zeros_like(c)
    
    # Pairwise contributions
    tri_i, tri_j = np.triu_indices(n, k=1)
    for idx in range(n_pairs):
        i, j = tri_i[idx], tri_j[idx]
        w = p_d[idx]
        d = dists[i, j]
        if d > 1e-15:
            dir_vec = (c[i] - c[j]) / d
            grad[i] += w * dir_vec / 2.0
            grad[j] -= w * dir_vec / 2.0
    
    # Wall contributions
    # w_vals is flattened wall_dists, shape (N*3,)
    for idx in range(n_walls):
        i = idx // 3
        k = idx % 3
        w = p_w[idx]
        # f = 0.5 - |c[i,k]|
        # df/dc[i,k] = -sign(c[i,k])
        if c[i, k] > 0:
            grad[i, k] -= w
        elif c[i, k] < 0:
            grad[i, k] += w
        # else: at 0, subgradient, use 0
    
    return r_soft, grad

def _init_grid():
    """5x5x2 grid initialization."""
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    
    return np.array(grid)

def _init_fcc():
    """FCC-like initialization."""
    # FCC lattice points in a cube
    # Generate points on FCC lattice and select 50 closest to center
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: (i,j,k) where i+j+k is even, scaled
                if (i + j + k) % 2 == 0:
                    points.append([i * 0.12, j * 0.12, k * 0.12])
    
    points = np.array(points)
    # Select 50 closest to origin
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:50]
    c = points[idx].copy()
    
    # Center and scale to fit in cube
    c -= np.mean(c, axis=0)
    max_ext = np.max(np.abs(c))
    if max_ext > 0.45:
        c *= 0.45 / max_ext
    
    return c

def _init_bcc():
    """BCC-like initialization."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 1:
                    points.append([i * 0.12, j * 0.12, k * 0.12])
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:50]
    c = points[idx].copy()
    
    c -= np.mean(c, axis=0)
    max_ext = np.max(np.abs(c))
    if max_ext > 0.45:
        c *= 0.45 / max_ext
    
    return c

def _init_random(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, (N, 3))
    return c

def _refine_smooth(c0, beta_start, beta_end, time_limit, seed=0):
    """
    Refine using smooth surrogate with annealing beta.
    Uses L-BFGS-B with time limit.
    """
    c = c0.copy()
    n = len(c)
    x0 = c.flatten()
    
    start_time = time.time()
    elapsed = 0.0
    
    # Annealing schedule
    n_iter_total = 5000
    beta_schedule = np.linspace(np.log(beta_start), np.log(beta_end), n_iter_total)
    
    def objective(x):
        c_x = x.reshape(n, 3)
        # Ensure inside cube
        c_x = np.clip(c_x, -0.499, 0.499)
        r, grad = _soft_min_radius_grad(c_x, beta_current[0])
        # Gradient of -r
        return -r, -grad.flatten()
    
    beta_current = [beta_start]
    
    # Use multiple phases with different beta
    n_phases = 5
    phase_betas = np.logspace(np.log10(beta_start), np.log10(beta_end), n_phases + 1)
    
    for phase in range(n_phases):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        beta_lo = phase_betas[phase]
        beta_hi = phase_betas[phase + 1]
        
        # For each phase, run L-BFGS-B with a few beta values
        n_sub = 3
        sub_betas = np.logspace(np.log10(beta_lo), np.log10(beta_hi), n_sub)
        
        for sub_beta in sub_betas:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break
            
            beta_current[0] = sub_beta
            
            # Time limit for this sub-problem
            sub_time = min(10.0, time_limit - elapsed - 5)
            if sub_time < 1:
                break
            
            def obj(x):
                c_x = x.reshape(n, 3)
                r, grad = _soft_min_radius_grad(c_x, sub_beta)
                return -r, -grad.flatten()
            
            # Bounds
            bounds = [(-0.499, 0.499)] * (n * 3)
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 10000,
                             'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < -_compute_radius(c0.reshape(n, 3)):
                    c = result.x.reshape(n, 3)
                    x0 = result.x
            except Exception:
                pass
            
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5:
                break
    
    return np.clip(c, -0.499, 0.499)

def _refine_exact(c0, time_limit, seed=0):
    """
    Exact refinement: push apart closest pairs, pull in wall-violating spheres.
    """
    c = c0.copy()
    n = len(c)
    start_time = time.time()
    
    step = 0.005
    decay = 0.999
    min_step = 1e-8
    rng = np.random.RandomState(seed)
    
    for it in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        d_pair = dists[i, j]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx
        d_wall = wall_dists[k, axis]
        
        if d_pair / 2.0 < d_wall:
            # Push i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = rng.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            # Max step to stay in bounds
            max_step = np.inf
            for coord in range(3):
                if direction[coord] > 0:
                    space_i = 0.499 - c[i][coord]
                    if space_i < 0: space_i = 0
                    max_step = min(max_step, space_i / direction[coord])
                    space_j = c[j][coord] + 0.499
                    if space_j < 0: space_j = 0
                    max_step = min(max_step, space_j / direction[coord])
                elif direction[coord] < 0:
                    space_i = c[i][coord] + 0.499
                    if space_i < 0: space_i = 0
                    max_step = min(max_step, space_i / (-direction[coord]))
                    space_j = 0.499 - c[j][coord]
                    if space_j < 0: space_j = 0
                    max_step = min(max_step, space_j / (-direction[coord]))
            
            actual_step = min(step, max_step)
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            # Pull k inward
            if c[k, axis] > 0:
                dir_val = -1.0
            else:
                dir_val = 1.0
            actual_step = min(step, 0.499 - abs(c[k, axis]))
            if actual_step > 1e-12:
                c[k, axis] += dir_val * actual_step
        
        step = max(min_step, step * decay)
        
        # Small noise
        if it % 500 == 0 and step > 1e-5:
            noise = rng.normal(0, step * 0.05, c.shape)
            c = np.clip(c + noise, -0.499, 0.499)
    
    return c

def _final_polish(c, time_limit):
    """Final exact polish with very small steps."""
    c = c.copy()
    start_time = time.time()
    
    step = 0.001
    decay = 0.999
    min_step = 1e-10
    
    for it in range(50000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i > j:
            i, j = j, i
        d_pair = dists[i, j]
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx
        d_wall = wall_dists[k, axis]
        
        if d_pair / 2.0 < d_wall:
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                continue
            direction /= norm
            
            max_step = np.inf
            for coord in range(3):
                if direction[coord] > 0:
                    max_step = min(max_step, (0.499 - c[i][coord]) / direction[coord])
                    max_step = min(max_step, (c[j][coord] + 0.499) / direction[coord])
                elif direction[coord] < 0:
                    max_step = min(max_step, (c[i][coord] + 0.499) / (-direction[coord]))
                    max_step = min(max_step, (0.499 - c[j][coord]) / (-direction[coord]))
            
            actual_step = min(step, max_step)
            if actual_step > 1e-12:
                c[i] += direction * actual_step
                c[j] -= direction * actual_step
        else:
            if c[k, axis] > 0:
                dir_val = -1.0
            else:
                dir_val = 1.0
            actual_step = min(step, 0.499 - abs(c[k, axis]))
            if actual_step > 1e-12:
                c[k, axis] += dir_val * actual_step
        
        step = max(min_step, step * decay)
    
    return c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    # Multiple initializations
    inits = [
        (_init_grid, "grid"),
        (_init_fcc, "fcc"),
        (_init_bcc, "bcc"),
        (_init_random, "rand"),
    ]
    
    best_c = None
    best_r = -np.inf
    
    # Time allocation
    # Phase 1: Smooth refinement on each init (60s total)
    # Phase 2: Exact refinement on best (50s)
    # Phase 3: Final polish (30s)
    
    phase1_time = 60.0
    phase2_time = 50.0
    phase3_time = 30.0
    
    # Phase 1: Smooth refinement
    t_per_init = phase1_time / len(inits)
    
    for init_func, name in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - phase2_time - phase3_time - 5:
            break
        
        t_limit = min(t_per_init, deadline - elapsed - phase2_time - phase3_time - 5)
        if t_limit < 2:
            break
        
        c0 = init_func()
        c_refined = _refine_smooth(c0, beta_start=10.0, beta_end=500.0, 
                                   time_limit=t_limit, seed=42)
        r = _compute_radius(c_refined)
        
        if r > best_r:
            best_r = r
            best_c = c_refined.copy()
    
    # Phase 2: Exact refinement on best
    if best_c is not None:
        elapsed = time.time() - start_time
        t_limit = min(phase2_time, deadline - elapsed - phase3_time - 2)
        if t_limit > 2:
            c_refined = _refine_exact(best_c, t_limit, seed=123)
            r = _compute_radius(c_refined)
            if r > best_r:
                best_r = r
                best_c = c_refined.copy()
    
    # Phase 3: Final polish
    if best_c is not None:
        elapsed = time.time() - start_time
        t_limit = min(phase3_time, deadline - elapsed - 1)
        if t_limit > 1:
            c_polished = _final_polish(best_c, t_limit)
            r = _compute_radius(c_polished)
            if r > best_r:
                best_r = r
                best_c = c_polished.copy()
    
    if best_c is None:
        best_c = _init_grid()
    
    # Ensure validity
    if not np.all(np.isfinite(best_c)):
        best_c = _init_grid()
    
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c

# EVOLVE-BLOCK-END
