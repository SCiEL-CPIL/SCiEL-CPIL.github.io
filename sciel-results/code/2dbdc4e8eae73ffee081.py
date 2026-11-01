"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Epigraph formulation with smooth constraints, repulsion annealing,
Voronoi Lloyd's algorithm, and multi-stage gradient polishing.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Voronoi

N = 50

# Precompute upper triangle indices
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)


def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)


def _epigraph_objective(x, t, alpha=20.0):
    """
    Epigraph formulation: minimize -t subject to all constraints.
    We use a smooth approximation: minimize -t + penalty * sum of soft violations.
    Actually, we maximize t subject to:
      dist(i,j)/2 >= t  for all pairs
      0.5 - |c_ik| >= t  for all walls
    
    Using log-sum-exp soft penalty for violations.
    """
    c = x.reshape(N, 3)
    
    # Pair distances
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0 - t
    
    # Wall distances
    wall_vals = 0.5 - np.abs(c) - t
    
    # Soft penalty for violations (negative values indicate violation)
    # Use log-sum-exp of exp(alpha * violation) where violation < 0
    # We want to penalize when val < 0
    # Use: -log(sum(exp(-alpha * val))) approximates min(val) with smoothing
    # But we want to penalize violations, so:
    # penalty = log(sum(exp(-alpha * val))) for val where val < 0
    
    # Actually, let's use a simpler approach:
    # objective = -t + beta * (sum of max(0, -val)^2 for all constraints)
    # But that's not smooth. Let's use smooth max.
    
    # Smooth max(0, x) ≈ log(1 + exp(alpha*x))/alpha
    def smooth_max(x, alpha):
        return np.logaddexp(0, alpha * x) / alpha
    
    pair_pen = np.sum(smooth_max(-pair_vals, alpha))
    wall_pen = np.sum(smooth_max(-wall_vals, alpha))
    
    return -t + 10.0 * (pair_pen + wall_pen)


def _epigraph_grad(x, t, alpha=20.0):
    """Gradient of the epigraph objective."""
    c = x.reshape(N, 3)
    grad_c = np.zeros_like(c)
    
    # Pair distances
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0 - t
    
    # Wall distances
    wall_vals = 0.5 - np.abs(c) - t
    
    # Smooth max gradient: sigmoid(alpha * x)
    def smooth_max_grad(x, alpha):
        return 1.0 / (1.0 + np.exp(-alpha * x))
    
    # Pair penalty gradient
    pair_grad_factor = smooth_max_grad(-pair_vals, alpha)
    # d(pair_val)/d(c_i) = 0.5 * (c_i - c_j) / dist
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    for idx in range(_NUM_PAIRS):
        i, j = _iu[idx], _ju[idx]
        f = 10.0 * pair_grad_factor[idx] * 0.5
        grad_c[i] += f * unit_vecs[idx]
        grad_c[j] -= f * unit_vecs[idx]
    
    # Wall penalty gradient
    wall_grad_factor = smooth_max_grad(-wall_vals, alpha)
    signs = np.sign(c)
    grad_c += 10.0 * wall_grad_factor * signs
    
    # Gradient of -t is 0 w.r.t. c, but t is a separate variable
    # We'll handle t separately
    
    return grad_c.ravel()


def _epigraph_optimize(c_init, t_init, deadline, alpha=20.0):
    """
    Optimize both centers and t using the epigraph formulation.
    """
    c = c_init.copy()
    t = t_init
    best_c = c.copy()
    best_t = t
    best_r = _compute_radius(c)
    
    # We'll do a few rounds with increasing alpha
    alphas = [10.0, 20.0, 40.0, 80.0]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        # Flatten: [c (N*3), t]
        x0 = np.concatenate([c.flatten(), [t]])
        
        def obj(x):
            c_x = x[:N*3].reshape(N, 3)
            t_x = x[N*3]
            # Clip to valid range
            c_x = np.clip(c_x, -0.4999, 0.4999)
            t_x = min(t_x, 0.49)
            
            diff = c_x[_iu] - c_x[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
            pair_vals = dists / 2.0 - t_x
            
            wall_vals = 0.5 - np.abs(c_x) - t_x
            
            def smooth_max(x, a):
                return np.logaddexp(0, a * x) / a
            
            pair_pen = np.sum(smooth_max(-pair_vals, alpha))
            wall_pen = np.sum(smooth_max(-wall_vals, alpha))
            
            return -t_x + 10.0 * (pair_pen + wall_pen)
        
        def grad(x):
            c_x = x[:N*3].reshape(N, 3)
            t_x = x[N*3]
            c_x = np.clip(c_x, -0.4999, 0.4999)
            t_x = min(t_x, 0.49)
            
            diff = c_x[_iu] - c_x[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
            pair_vals = dists / 2.0 - t_x
            
            wall_vals = 0.5 - np.abs(c_x) - t_x
            
            def smooth_max_grad(x, a):
                return 1.0 / (1.0 + np.exp(-a * x))
            
            grad_c = np.zeros_like(c_x)
            
            pair_grad_factor = smooth_max_grad(-pair_vals, alpha)
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            for idx in range(_NUM_PAIRS):
                i, j = _iu[idx], _ju[idx]
                f = 10.0 * pair_grad_factor[idx] * 0.5
                grad_c[i] += f * unit_vecs[idx]
                grad_c[j] -= f * unit_vecs[idx]
            
            wall_grad_factor = smooth_max_grad(-wall_vals, alpha)
            signs = np.sign(c_x)
            grad_c += 10.0 * wall_grad_factor * signs
            
            # Gradient of -t
            grad_t = -1.0
            # Plus gradient from penalty terms w.r.t. t
            # d/dt of smooth_max(-pair_val, alpha) = -smooth_max_grad(-pair_val, alpha)
            # d/dt of smooth_max(-wall_val, alpha) = -smooth_max_grad(-wall_val, alpha)
            grad_t += 10.0 * (-np.sum(pair_grad_factor) - np.sum(wall_grad_factor))
            
            return np.concatenate([grad_c.ravel(), [grad_t]])
        
        bounds = [(-0.4999, 0.4999)] * (N * 3) + [(0.0, 0.49)]
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
            )
            
            c_new = res.x[:N*3].reshape(N, 3)
            t_new = res.x[N*3]
            
            c_new = np.clip(c_new, -0.4999, 0.4999)
            t_new = min(t_new, 0.49)
            
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                best_t = t_new
                c = c_new.copy()
                t = t_new
                
        except Exception:
            continue
    
    return best_c, best_r


def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temp = 0.04
    decay = 0.997
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2 * r_target, 1.0 / (dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        forces = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _iu[idx], _ju[idx]
            f = force_scale[idx] * diff[idx]
            forces[i] += f
            forces[j] -= f
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_force = np.zeros_like(c)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < r_target:
                    wall_force[i, k] = np.sign(c[i, k]) * (r_target - wall_dists[i, k]) * 3.0
        
        total_force = forces + wall_force
        
        # Random perturbation
        noise = np.random.normal(0.0, temp, size=c.shape)
        
        c_new = c + 0.008 * total_force + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Evaluate
        r_new = _compute_radius(c_new)
        r_curr = _compute_radius(c)
        
        # Accept if better or with probability
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / max(temp, 1e-6)):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.0005:
            temp = 0.0005
    
    return best_c, best_r


def _lloyd_step(c, iterations=20, clip_margin=0.001):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell.
    """
    c = c.copy()
    for _ in range(iterations):
        try:
            vor = Voronoi(c)
        except Exception:
            break
        
        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                new_c[i] = c[i] * 0.95
                continue
            
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_c[i] = c[i] * 0.95
                continue
            
            verts = np.clip(verts, -0.5 + clip_margin, 0.5 - clip_margin)
            centroid = np.mean(verts, axis=0)
            
            new_c[i] = 0.8 * c[i] + 0.2 * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        c = new_c
    
    return c


def _smooth_polish(c_init, deadline, t_start=0.03, t_end=0.0005):
    """
    Polish using gradient ascent on softmin radius with decreasing temperature.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 8
    temps = np.geomspace(t_start, t_end, n_temps)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        def obj(x):
            c_x = x.reshape(N, 3)
            diff = c_x[_iu] - c_x[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
            pair_vals = dists / 2.0
            wall_vals = 0.5 - np.abs(c_x)
            
            # Softmin via log-sum-exp
            all_vals = np.concatenate([pair_vals, wall_vals.flatten()])
            max_val = np.max(all_vals)
            log_sum = max_val + np.log(np.sum(np.exp(-all_vals / t)))
            softmin_val = t * log_sum - t * max_val  # This is -t*log(sum(exp(-val/t)))
            # Actually: softmin = -t * log(sum(exp(-val/t)))
            # Let me redo:
            shifted = -all_vals / t
            max_shifted = np.max(shifted)
            log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            softmin_val = -t * log_sum
            return -softmin_val
        
        def grad(x):
            c_x = x.reshape(N, 3)
            diff = c_x[_iu] - c_x[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
            pair_vals = dists / 2.0
            wall_vals = 0.5 - np.abs(c_x)
            
            all_vals = np.concatenate([pair_vals, wall_vals.flatten()])
            shifted = -all_vals / t
            max_shifted = np.max(shifted)
            weights = np.exp(shifted - max_shifted)
            weights /= np.sum(weights)
            
            grad_c = np.zeros_like(c_x)
            
            # Pair contributions
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            for idx in range(_NUM_PAIRS):
                i, j = _iu[idx], _ju[idx]
                w = weights[idx]
                f = -w * 0.5
                grad_c[i] += f * unit_vecs[idx]
                grad_c[j] -= f * unit_vecs[idx]
            
            # Wall contributions
            wall_weights = weights[_NUM_PAIRS:]
            signs = np.sign(c_x)
            grad_c += wall_weights.reshape(N, 3) * signs
            
            return grad_c.ravel()
        
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 70}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
    
    return best_c, best_r


def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            
            for step in (0.002, 0.004, 0.008, 0.015):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            # Pair binding
            min_idx = np.argmin(dists)
            i, j = _iu[min_idx], _ju[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.004, 0.008, 0.015):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    
    return best_c, best_r


def _seed_fcc():
    """Generate an FCC-inspired initial configuration."""
    a = 1.0 / 4.0
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                if i < 3:
                    points.append((i*a - 0.5 + a/2, j*a - 0.5, k*a - 0.5))
                if j < 3:
                    points.append((i*a - 0.5, j*a - 0.5 + a/2, k*a - 0.5))
                if k < 3:
                    points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5 + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])


def _seed_bcc():
    """Generate a BCC-inspired initial configuration."""
    a = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                points.append((i*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])


def _seed_grid():
    """Generate a grid-based initial configuration."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx] + np.random.normal(0, 0.01, size=(N, 3))


def _seed_random():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.4, 0.4, size=(N, 3))


def _seed_shell_core():
    """Generate a shell-core initial configuration."""
    c = np.zeros((N, 3))
    core = np.random.uniform(-0.15, 0.15, size=(8, 3))
    c[:8] = core
    shell = np.random.uniform(-0.45, 0.45, size=(42, 3))
    c[8:] = shell
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    seeds = [
        _seed_fcc(),
        _seed_bcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        _seed_random(),
    ]
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Reserve time for final polish
    final_time_budget = 35.0
    search_time = (deadline - start_time) - final_time_budget
    time_per_seed = search_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - final_time_budget - 5.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.13)
        
        if r_anneal > best_r:
            best_r = r_anneal
            best_c = c_anneal.copy()
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 2: Lloyd's algorithm
        c_lloyd = _lloyd_step(c_anneal, iterations=15)
        c_lloyd = np.clip(c_lloyd, -0.4999, 0.4999)
        r_lloyd = _compute_radius(c_lloyd)
        
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 3: Smooth polish
        c_polish, r_polish = _smooth_polish(c_lloyd, seed_deadline, t_start=0.02, t_end=0.001)
        
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 4: Epigraph optimization
        c_epi, r_epi = _epigraph_optimize(c_polish, best_r * 0.95, seed_deadline)
        
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
    
    # Final polish on the global best
    final_deadline = deadline - 5.0
    
    if time.time() < final_deadline:
        # Multi-stage final polish
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
        
        # Epigraph polish
        if time.time() < final_deadline - 3.0:
            c_epi, r_epi = _epigraph_optimize(best_c, best_r * 0.98, final_deadline, alpha=40.0)
            if r_epi > best_r:
                best_r = r_epi
                best_c = c_epi.copy()
        
        # Local search
        if time.time() < final_deadline - 2.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
        
        # Ultra-fine final polish
        if time.time() < final_deadline - 1.0:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.001, t_end=0.00003)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
    
    return best_c
