import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _compute_radius_fast(p, prev_dists=None):
    """Compute radius with optional cached distance matrix for speed."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    if np.any(np.abs(p) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint - recompute distances
    dists = cdist(p, p, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _metropolis_mc(p0, target_r, n_steps=8000, time_limit=None, start_time=None):
    """Metropolis Monte Carlo to escape local minima."""
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    step_size = 0.012
    T = 0.004  # Temperature
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Pick a random point and axis
        i = np.random.randint(N)
        axis = np.random.randint(3)
        delta = np.random.normal(0, step_size)
        
        p_trial = p.copy()
        p_trial[i, axis] += delta
        p_trial = np.clip(p_trial, -0.5, 0.5)
        
        r_trial = _compute_radius(p_trial)
        
        # Accept if better, or with probability exp(-(r_trial - current_r) / T)
        if r_trial > current_r:
            p = p_trial
            current_r = r_trial
            if current_r > best_r:
                best_r = current_r
                best_p = p_trial.copy()
        else:
            delta_r = current_r - r_trial
            if delta_r < 50 * T:  # Avoid overflow
                prob = np.exp(-delta_r / T)
                if np.random.random() < prob:
                    p = p_trial
                    current_r = r_trial
        
        # Anneal temperature more aggressively
        T = max(0.0008, T * 0.999)
    
    return best_p, best_r

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    if np.any(np.abs(p) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _pair_dist_matrix(p):
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    return np.linalg.norm(diffs, axis=2)

def _softmin_radius_and_grad(p, beta=50.0):
    """
    Smooth approximation of min radius using softmin, with analytic gradient.
    Vectorized for speed.
    
    Returns (softmin_value, gradient_wrt_p) where gradient is d(softmin)/dp.
    """
    # Boundary distances: b_i = 0.5 - max_j |p_ij|
    abs_p = np.abs(p)
    max_abs = np.max(abs_p, axis=1)  # (N,)
    b = 0.5 - max_abs  # (N,)
    
    # Pairwise distances / 2 using cdist for efficiency
    dists = cdist(p, p, metric='euclidean')  # (N,N)
    np.fill_diagonal(dists, np.inf)
    pd = dists / 2.0  # (N,N)
    
    # Collect all constraints: N boundary + N*(N-1)/2 pairwise
    iu = np.triu_indices(N, k=1)
    pd_upper = pd[iu]  # (N*(N-1)/2,)
    
    all_constraints = np.concatenate([b, pd_upper])  # (N + N*(N-1)/2,)
    
    # Softmin: f = min_val - (1/beta) * log(sum(exp(-beta * (x - min_val))))
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val  # >= 0
    exp_vals = np.exp(-beta * shifted)  # in (0, 1]
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    softmin_val = min_val - log_sum / beta
    
    # Gradient: d(softmin)/dp_k = sum_j (exp_j / sum_exp) * d x_j / dp_k
    weights = exp_vals / sum_exp  # (N + N*(N-1)/2,)
    
    grad = np.zeros_like(p)
    
    # Boundary part: x_j = b_j = 0.5 - max_k |p_jk|
    # d b_j / dp_jk = -sign(p_jk) if k is the argmax axis for point j
    # Note: if multiple axes tie, use the first one (argmax behavior)
    boundary_weights = weights[:N]
    max_axes = np.argmax(abs_p, axis=1)  # (N,)
    # Vectorized boundary gradient
    signs = np.sign(p[np.arange(N), max_axes])
    # Handle zero case: if p[j,k]==0, gradient is 0
    signs[signs == 0] = 0
    boundary_grad = np.zeros_like(p)
    boundary_grad[np.arange(N), max_axes] = -boundary_weights * signs
    grad += boundary_grad
    
    # Pairwise part: vectorized
    pair_weights = weights[N:]  # (N*(N-1)/2,)
    if np.any(pair_weights > 1e-15):
        i_idx = iu[0]
        j_idx = iu[1]
        # Compute unit vectors for all pairs at once
        diffs = p[i_idx] - p[j_idx]  # (M, 3)
        dists_pair = np.linalg.norm(diffs, axis=1)  # (M,)
        # Avoid division by zero
        safe_dists = np.where(dists_pair > 1e-12, dists_pair, 1.0)
        unit_vecs = diffs / safe_dists[:, np.newaxis]  # (M, 3)
        
        # Accumulate gradients: for each pair (i,j) with weight w:
        # grad[i] += w * 0.5 * unit_vec
        # grad[j] += w * (-0.5) * unit_vec
        w_scaled = pair_weights * 0.5
        # Use np.add.at for accumulation
        np.add.at(grad, i_idx, w_scaled[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -w_scaled[:, np.newaxis] * unit_vecs)
    
    return softmin_val, grad

def _smooth_obj_and_grad(p, beta=50.0):
    """Objective to maximize: negative softmin radius, with gradient."""
    val, grad = _softmin_radius_and_grad(p, beta)
    return -val, -grad

def _lloyd_step(p, n_steps=1, time_limit=None, start_time=None):
    """
    Perform Lloyd's algorithm steps to equalize Voronoi cell sizes.
    This pushes points away from crowded areas and towards empty areas.
    Uses scipy.spatial.Voronoi.
    """
    from scipy.spatial import Voronoi, QhullError
    
    p = p.copy()
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            vor = Voronoi(p)
        except QhullError:
            break
            
        new_p = p.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded region: nudge slightly toward center
                new_p[i] = 0.95 * p[i] + 0.05 * np.array([0.0, 0.0, 0.0])
            else:
                vertices = vor.vertices[region]
                centroid = np.mean(vertices, axis=0)
                new_p[i] = centroid
        
        margin = 1e-6
        new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)
        
        # Push points near walls back in
        dist_to_wall = 0.5 - np.max(np.abs(new_p), axis=1)
        close_to_wall = dist_to_wall < 0.02
        if np.any(close_to_wall):
            center = np.zeros(3)
            new_p[close_to_wall] += 0.03 * (center - new_p[close_to_wall])
            new_p = np.clip(new_p, -0.5 + margin, 0.5 - margin)

        p = new_p
        
    return p

def _repulsion_anneal(p0, target_r, n_steps=200, time_limit=None, start_time=None):
    """Repulsion-based local search to push points apart while respecting target radius."""
    p = p0.copy()
    limits = 0.5 - target_r
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = np.zeros_like(p)
        
        # Boundary forces
        for axis in range(3):
            over_pos = p[:, axis] > limits
            over_neg = p[:, axis] < -limits
            forces[over_pos, axis] -= 1.0
            forces[over_neg, axis] += 1.0
        
        # Pairwise repulsion
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        mask = dists < 2 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            unit_vecs = unit_vecs * mask[:, :, np.newaxis]
            pair_forces = np.sum(unit_vecs, axis=1)
            forces += pair_forces
        
        # Adaptive step size
        step_size = 0.02 * (1.0 - 0.5 * step / n_steps)
        p += step_size * forces
        
        # Clamp to cube
        p = np.clip(p, -0.5, 0.5)
    
    return p

def _seed_fcc():
    """Generate FCC lattice points and select the most compact cluster."""
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]].copy()

def _seed_grid():
    """Generate a 3D grid of points."""
    # 4x4x4 = 64 points, take 50
    coords = np.linspace(-0.4, 0.4, 4)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    grid -= np.mean(grid, axis=0)
    
    dists = np.linalg.norm(grid, axis=1)
    idx = np.argsort(dists)
    
    return grid[idx[:N]].copy()

def _seed_random(seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(-0.4, 0.4, size=(N, 3))
    return p

def _local_search(p0, time_limit, start_time):
    """
    Combined local search:
    1. Lloyd steps for global equalization.
    2. Multi-scale soft-min gradient polishing with analytic gradients.
    3. Metropolis MC to escape local minima.
    4. Repulsion annealing if stuck.
    """
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    # Multi-scale beta values for soft-min - start coarser, refine later
    betas = [10.0, 30.0, 100.0, 300.0, 500.0, 1000.0, 2000.0]
    
    # Track iterations to increase polish frequency
    iter_count = 0
    
    # Main loop
    while time.time() - start_time < time_limit:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3.0:
            break
        iter_count += 1
        
        # 1. Lloyd Step (less frequent, more expensive) - only early on
        if iter_count < 5 and np.random.random() < 0.3:
            p_lloyd = _lloyd_step(p, n_steps=2, time_limit=remaining, start_time=start_time)
            r_lloyd = _compute_radius(p_lloyd)
            
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_p = p_lloyd.copy()
                p = p_lloyd
            else:
                p = best_p.copy()
        
        # 2. Soft-min Polish with analytic gradients
        improved_in_polish = False
        # Use more aggressive betas as we progress
        if iter_count < 3:
            active_betas = betas[:3]
        elif iter_count < 6:
            active_betas = betas[2:5]
        else:
            active_betas = betas[4:]
            
        for beta in active_betas:
            if time.time() - start_time > time_limit:
                break
            
            def obj_and_grad(x, _beta=beta):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=_beta)
                return val, grad
            
            # Use L-BFGS-B with analytic gradient
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                max_iter = 100 if beta < 500 else 150
                gtol = 1e-9 if beta < 500 else 1e-10
                result = minimize(obj_and_grad, p.flatten(), method='L-BFGS-B',
                                 jac=True, bounds=bounds,
                                 options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 25})
                p_smooth = result.x.reshape(N, 3)
                r_smooth = _compute_radius(p_smooth)
                
                if r_smooth > best_r + 1e-12:
                    best_r = r_smooth
                    best_p = p_smooth.copy()
                    p = p_smooth
                    improved_in_polish = True
            except Exception:
                pass
        
        # 3. If no improvement, try Metropolis MC
        if not improved_in_polish:
            mc_time = min(remaining * 0.25, 8.0)
            p_mc, r_mc = _metropolis_mc(best_p, best_r, n_steps=4000, 
                                        time_limit=time.time() + mc_time,
                                        start_time=start_time)
            if r_mc > best_r + 1e-12:
                best_r = r_mc
                best_p = p_mc.copy()
                p = p_mc
            else:
                # 4. Try repulsion anneal
                p_rep = _repulsion_anneal(best_p, best_r, n_steps=60, 
                                          time_limit=remaining,
                                          start_time=start_time)
                r_rep = _compute_radius(p_rep)
                if r_rep > best_r + 1e-12:
                    best_r = r_rep
                    best_p = p_rep.copy()
                    p = p_rep
                else:
                    if time.time() - start_time > time_limit * 0.85:
                        break
                    # Small random perturbation to escape local minima
                    p = best_p + np.random.normal(0, 0.002, size=(N, 3))
                    p = np.clip(p, -0.5, 0.5)
    
    return best_p, best_r

def _coordinate_descent_polish(p, target_r, max_iters=200, time_limit=None, start_time=None):
    """
    Coordinate descent to push points apart while maintaining radius >= target_r.
    For each point, try small moves in each coordinate to increase the minimum radius.
    Uses adaptive step sizes.
    """
    p = p.copy()
    current_r = _compute_radius(p)
    
    # Adaptive step sizes - start larger, shrink over iterations
    base_steps = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Shuffle point order for diversity
        order = np.random.permutation(N)
        
        # Shrink steps as we progress
        step_scale = max(0.3, 1.0 - 0.5 * it / max_iters)
        steps = [s * step_scale for s in base_steps]
        
        for i in order:
            best_move = None
            best_r_candidate = current_r
            
            # Try perturbations in each direction with adaptive sizes
            for axis in range(3):
                for delta in steps + [-s for s in steps]:
                    p_trial = p.copy()
                    p_trial[i, axis] += delta
                    p_trial = np.clip(p_trial, -0.5, 0.5)
                    r_trial = _compute_radius(p_trial)
                    if r_trial > best_r_candidate + 1e-13:
                        best_r_candidate = r_trial
                        best_move = (axis, delta)
            
            if best_move is not None:
                p[i, best_move[0]] += best_move[1]
                p = np.clip(p, -0.5, 0.5)
                current_r = _compute_radius(p)
                improved = True
        
        if not improved:
            break
    
    return p, current_r

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin from 150s
    
    # Try multiple seeds and pick the best
    seeds = [
        ('fcc', _seed_fcc),
        ('grid', _seed_grid),
        ('random', _seed_random),
    ]
    
    best_p = None
    best_r = -1
    
    # Allocate time per seed - give more time to early seeds, less to later ones
    time_allocations = [0.50, 0.28, 0.22]
    available_time = total_time_limit - 12.0
    
    for i, (seed_name, seed_func) in enumerate(seeds):
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 10.0:
            break
        
        p0 = seed_func()
        
        # Scale to fit in cube with some margin
        max_coord = np.max(np.abs(p0))
        if max_coord > 0:
            scale = 0.42 / max_coord
            p0 = p0 * scale
        
        # Local search
        seed_time_limit = time_allocations[i] * available_time
        seed_time_limit = min(seed_time_limit, total_time_limit - 10.0 - elapsed)
        if seed_time_limit < 5.0:
            break
        
        p_improved, r_improved = _local_search(p0, seed_time_limit, start_time)
        
        if r_improved > best_r:
            best_r = r_improved
            best_p = p_improved
    
    # Final high-precision polish with multiple beta levels
    if best_p is not None and time.time() - start_time < total_time_limit - 6.0:
        p = best_p.copy()
        
        for beta in [1000.0, 2000.0, 5000.0]:
            if time.time() - start_time > total_time_limit - 4.0:
                break
            
            def obj_and_grad_final(x, _beta=beta):
                val, grad = _smooth_obj_and_grad(x.reshape(N, 3), beta=_beta)
                return val, grad
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                result = minimize(obj_and_grad_final, p.flatten(), method='L-BFGS-B',
                                 jac=True, bounds=bounds,
                                 options={'maxiter': 500, 'gtol': 1e-13, 'maxls': 40})
                p_smooth = result.x.reshape(N, 3)
                r_smooth = _compute_radius(p_smooth)
                
                if r_smooth > best_r:
                    best_r = r_smooth
                    best_p = p_smooth
                    p = p_smooth
            except Exception:
                pass
    
    # Coordinate descent polish for final squeeze
    if best_p is not None and time.time() - start_time < total_time_limit - 3.0:
        remaining = total_time_limit - 3.0 - (time.time() - start_time)
        p_cd, r_cd = _coordinate_descent_polish(best_p, best_r, max_iters=200, 
                                                  time_limit=remaining, start_time=start_time)
        if r_cd > best_r:
            best_r = r_cd
            best_p = p_cd
    
    # Final validation
    if best_p is not None:
        final_r = _compute_radius(best_p)
        if final_r > best_r:
            best_r = final_r
    
    return best_p
