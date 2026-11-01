"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-start with diverse seeds, Voronoi-based Lloyd equalization,
soft-min gradient polishing, Metropolis Monte Carlo, and local search.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp formulation.
    Returns (radius, gradient) where gradient is (N, 3).
    
    The softmin is: r = -t * log(sum(exp(-v_i/t))) where v_i are the constraint values.
    Gradient: dr/dx = (1/sum) * sum(exp(-v_i/t) * dv_i/dx)
    """
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    n_walls = N * 3
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    max_val = np.max(all_vals)
    
    # Weights proportional to exp(-v_i / t), normalized
    pair_weights = np.exp(-(pair_vals - max_val) / t)
    wall_weights = np.exp(-(wall_vals - max_val) / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        m = min(np.min(pair_vals), np.min(wall_vals))
        return m, np.zeros_like(c)
        
    # softmin value
    softmin_val = max_val - t * np.log(total_sum)
    
    # Normalized weights (sum to 1)
    pair_norm = pair_weights / total_sum
    wall_norm = wall_weights / total_sum
    
    grad_c = np.zeros_like(c)
    
    # Pair contributions: d(dist/2)/d(c_i) = 0.5 * (c_i - c_j)/|c_i - c_j|
    # So dr/dc_i = sum_j w_ij * 0.5 * (c_i - c_j)/|c_i - c_j| for pairs where i is first
    # and dr/dc_j = sum_i w_ij * (-0.5) * (c_i - c_j)/|c_i - c_j|
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Each pair (i,j) contributes:
    #   to c[i]: +0.5 * w_ij * unit_vec
    #   to c[j]: -0.5 * w_ij * unit_vec
    # where unit_vec = (c_i - c_j) / |c_i - c_j|
    # But we want to MAXIMIZE r, so gradient of r w.r.t. c[i] is +0.5*w*unit_vec
    # Wait: r = min(dist/2, wall). For pair constraint, v = dist/2.
    # d(dist/2)/d(c_i) = 0.5 * (c_i - c_j)/dist
    # So dr/dc_i = w_i * 0.5 * (c_i-c_j)/dist (for the pair term)
    # We want to move in direction of increasing r, so gradient ascent uses +grad.
    
    grad_contrib = 0.5 * pair_norm[:, np.newaxis] * unit_vecs
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        grad_c[i] += grad_contrib[idx]
        grad_c[j] -= grad_contrib[idx]
    
    # Wall contributions: d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
    # dr/dc_ik = w_ik * (-sign(c_ik))
    # To maximize r, we move in direction +grad, so we want to decrease |c_ik|
    # which means moving toward center: direction = -sign(c_ik)
    signs = np.sign(c)
    grad_wall_contrib = -wall_norm * signs
    grad_c += grad_wall_contrib
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, n_steps=2000, t_start=0.5, t_end=0.01):
    """
    Repulsion annealing: minimize sum of inverse distances between all pairs
    plus wall penalties, with decreasing temperature for step size.
    This is faster than Lloyd for initial equalization.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / max(n_steps - 1, 1))
        step_scale = 0.01 * t / t_start
        
        # Compute forces: repulsion between pairs + wall attraction
        forces = np.zeros_like(c)
        
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        
        # Repulsion force: F = k / dist^2 * unit_vec (pushes apart)
        # Use a softened potential to avoid singularities
        k = 0.1
        repulsion = k / (dists ** 2 + 1e-6)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[:, np.newaxis]
        unit_vecs[np.isnan(unit_vecs)] = 0.0
        
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            f = repulsion[idx] * unit_vecs[idx]
            forces[i] += f
            forces[j] -= f
        
        # Wall attraction: pull toward center if too close to wall
        wall_margin = 0.5 - np.abs(c)
        wall_force = np.zeros_like(c)
        close_to_wall = wall_margin < 0.1
        wall_force[close_to_wall] = -np.sign(c[close_to_wall]) * (0.1 - wall_margin[close_to_wall]) * 2.0
        forces += wall_force
        
        # Update with adaptive step
        c_new = c + step_scale * forces
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        
        c = c_new
    
    return best_c, best_r

def _lloyd_step(c, iterations=40, clip_margin=0.001):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell,
    clipped to the cube boundary. This equalizes clearance around each sphere.
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
                new_c[i] = c[i] * 0.9
                continue
            
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_c[i] = c[i] * 0.9
                continue
            
            verts = np.clip(verts, -0.5 + clip_margin, 0.5 - clip_margin)
            centroid = np.mean(verts, axis=0)
            new_c[i] = 0.6 * c[i] + 0.4 * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        c = new_c
    
    return c

def _lloyd_optimize(c_init, deadline):
    """
    Run Lloyd's algorithm followed by smooth polishing.
    """
    c = _lloyd_step(c_init, iterations=30)
    c = np.clip(c, -0.4999, 0.4999)
    r = _compute_radius(c)
    
    c, r = _smooth_polish(c, deadline, t_start=0.02, t_end=0.001)
    
    return c, r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0005):
    """
    Polish the configuration using gradient ascent on the softmin radius
    with decreasing temperature for increasingly sharp approximation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # More temperature steps for finer control
    n_temps = 15
    temps = np.geomspace(t_start, t_end, n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
        
        # Use more iterations for later (sharper) temperatures
        maxiter = 300 if ti < n_temps // 2 else 600
        
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
            
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj, x0, jac=grad,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
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

def _metropolis_mc(c_init, deadline, n_steps=2000):
    """
    Metropolis Monte Carlo to escape local minima.
    Uses both single-point moves and pair swaps.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temp = 0.03
    step_size = 0.02
    iu = np.triu_indices(N, k=1)
    
    # Precompute for efficiency
    n_steps_actual = 0
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        n_steps_actual += 1
        
        # Pick a random point to move
        i = np.random.randint(N)
        old_pos = c[i].copy()
        
        # Occasionally try a swap with another point
        if np.random.random() < 0.1:
            j = np.random.randint(N)
            if j == i:
                continue
            old_j = c[j].copy()
            c[i], c[j] = old_j.copy(), old_pos.copy()
            c[i] = np.clip(c[i], -0.4999, 0.4999)
            c[j] = np.clip(c[j], -0.4999, 0.4999)
            r_after = _compute_radius(c)
            r_before = best_r
            
            if r_after > r_before or np.random.random() < np.exp((r_after - r_before) / temp):
                if r_after > best_r:
                    best_r = r_after
                    best_c = c.copy()
            else:
                c[i], c[j] = old_pos, old_j
        else:
            c[i] = old_pos + np.random.normal(0, step_size, size=3)
            c[i] = np.clip(c[i], -0.4999, 0.4999)
            r_after = _compute_radius(c)
            
            # Compute r_before more efficiently by tracking
            r_before = _compute_radius(c)
            
            if r_after > r_before or np.random.random() < np.exp((r_after - r_before) / temp):
                if r_after > best_r:
                    best_r = r_after
                    best_c = c.copy()
            else:
                c[i] = old_pos
        
        # Adaptive step size based on acceptance
        if r_after > best_r:
            step_size *= 1.02
        elif it % 100 == 0:
            step_size *= 0.98
        step_size = max(0.001, min(step_size, 0.05))
        
        temp *= 0.999
        temp = max(temp, 0.003)
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    Uses Delaunay adjacency to focus on topologically relevant pairs.
    """
    from scipy.spatial import Delaunay
    
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    
    # Build Delaunay adjacency for efficient neighbor lookup
    try:
        tri = Delaunay(c)
        # Build adjacency set from simplices
        adj = [set() for _ in range(N)]
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = simplex[i], simplex[j]
                    adj[a].add(b)
                    adj[b].add(a)
        has_adj = True
    except Exception:
        has_adj = False
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
            c = best_c.copy()
            r = best_r
             
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
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
            # Try to find the tightest pair among Delaunay neighbors first
            if has_adj:
                best_pair_idx = -1
                best_pair_dist = np.inf
                for i in range(N):
                    for j in adj[i]:
                        if j > i:
                            d = np.linalg.norm(c[i] - c[j])
                            if d < best_pair_dist:
                                best_pair_dist = d
                                best_pair_idx = i * N + j  # encode pair
                if best_pair_idx >= 0:
                    i = best_pair_idx // N
                    j = best_pair_idx % N
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
                    continue
            
            # Fallback: global minimum pair
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
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
    """
    Generate an FCC-like initial configuration.
    """
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
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx] + np.random.normal(0, 0.01, size=(N, 3))

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, size=(N, 3))

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    c = np.zeros((N, 3))
    core = np.random.uniform(-0.15, 0.15, size=(8, 3))
    c[:8] = core
    shell = np.random.uniform(-0.45, 0.45, size=(42, 3))
    c[8:] = shell
    return c

def _seed_bcc():
    """
    Generate a BCC-like initial configuration.
    """
    a = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2))
    # Add body centers
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        _seed_bcc(),
        _seed_random(),
    ]
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time: reserve 25% for final polish
    total_time = deadline - start_time
    final_time = total_time * 0.25
    search_time = total_time - final_time
    time_per_seed = search_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - final_time - 3.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Repulsion annealing (fast initial equalization)
        c_rep, r_rep = _repulsion_anneal(s, min(seed_start + time_per_seed * 0.3, seed_deadline), n_steps=1000)
        
        if r_rep > best_r:
            best_r = r_rep
            best_c = c_rep.copy()
        
        # Stage 2: Lloyd's algorithm for finer equalization
        c_lloyd = _lloyd_step(c_rep, iterations=15)
        c_lloyd = np.clip(c_lloyd, -0.4999, 0.4999)
        r_lloyd = _compute_radius(c_lloyd)
        
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
        
        # Stage 3: Smooth polish
        c_polish, r_polish = _smooth_polish(c_lloyd, seed_deadline, t_start=0.03, t_end=0.001)
        
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
        
        # Stage 4: Metropolis MC
        c_mc, r_mc = _metropolis_mc(c_polish, seed_deadline, n_steps=800)
        
        if r_mc > best_r:
            best_r = r_mc
            best_c = c_mc.copy()
        
        # Stage 5: Local search
        c_ls, r_ls = _local_search(c_mc, seed_deadline, max_iters=150)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    final_deadline = deadline - 3.0
    if time.time() < final_deadline:
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.003, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        if time.time() < final_deadline - 2.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=400)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        if time.time() < final_deadline - 1.5:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.0008, t_end=0.00002)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
        # One more local search pass after the final polish
        if time.time() < final_deadline - 1.0:
            c_ls2, r_ls2 = _local_search(best_c, final_deadline, max_iters=150)
            if r_ls2 > best_r:
                best_r = r_ls2
                best_c = c_ls2.copy()
                
    return best_c

# EVOLVE-BLOCK-END
