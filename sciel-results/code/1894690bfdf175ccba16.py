"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization with vectorized operations:
1. Structured initial configurations (FCC, grid, BCC, random)
2. Vectorized repulsion-based simulated annealing
3. Metropolis Monte Carlo for global search
4. Soft-min gradient polishing (log-sum-exp)
5. Local search on binding constraints
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration for 50 spheres.
    """
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0),
                                    (0.5, 0.5, 0),
                                    (0.5, 0, 0.5),
                                    (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_bcc():
    """
    Generate a BCC-inspired initial configuration.
    """
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    n_core = 20
    n_shell = N - n_core
    
    core = np.random.uniform(-0.25, 0.25, size=(n_core, 3))
    
    theta = np.random.uniform(0, 2 * np.pi, size=n_shell)
    phi = np.random.uniform(0, np.pi, size=n_shell)
    r = 0.35 + np.random.uniform(-0.05, 0.05, size=n_shell)
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    return np.vstack([core, shell])

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    Uses log-sum-exp for numerical stability.
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # For each pair (i,j), the gradient contribution to c[i] is -0.5 * w * (c[i]-c[j])/d
    # and to c[j] is +0.5 * w * (c[i]-c[j])/d
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] -= pair_grad[idx]
        grad_c[j] += pair_grad[idx]
    
    # Wall contributions
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
    """
    Vectorized repulsion-based simulated annealing.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces (vectorized)
            forces = np.zeros_like(c)
            
            # Pair repulsion
            diff = c[_iu] - c[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            for idx in range(_NUM_PAIRS):
                i, j = _PAIR_IDX[idx]
                forces[i] += pair_forces[idx]
                forces[j] -= pair_forces[idx]
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists * wall_dists)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.1, size=c.shape)
            c_new = c + forces * 0.01 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            # Accept if better or with probability exp(-delta_r / t)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(delta_r / t):
                c = c_new
    
    return best_c, best_r

def _metropolis_mc(c_init, deadline, n_steps=5000, t_start=0.05, t_end=0.001):
    """
    Metropolis Monte Carlo for global search.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.geomspace(t_start, t_end, 20)
    steps_per_temp = max(1, n_steps // len(temps))
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Propose random move: pick a random point and move it
            i = np.random.randint(N)
            move = np.random.normal(0, t * 0.5, size=3)
            c_new = c.copy()
            c_new[i] += move
            c_new[i] = np.clip(c_new[i], -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(delta_r / t):
                c = c_new
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0005):
    """
    Polish the configuration using gradient ascent on the softmin radius
    with decreasing temperature for increasingly sharp approximation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Geometric progression of temperatures
    n_temps = 12
    temps = np.geomspace(t_start, t_end, n_temps)
    
    for t in temps:
        if time.time() > deadline:
            break
            
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
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
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

def _local_search(c_init, deadline, max_iters=300):
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
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        # Find the binding constraint
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            # Direction away from wall
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving
            for step in (0.003, 0.006, 0.01, 0.02):
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
            i, j = _PAIR_IDX[min_idx]
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.003, 0.006, 0.01, 0.02):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_bcc())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())
    seeds.append(_seed_grid())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: give more time to final polish
    # Reserve 25% of time for final polish
    final_time = (deadline - start_time) * 0.25
    search_time = (deadline - start_time) - final_time
    time_per_seed = search_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - final_time - 5.0)
        
        if time.time() > seed_deadline:
            continue
            
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline)
        
        if time.time() > seed_deadline:
            if r_anneal > best_r:
                best_r = r_anneal
                best_c = c_anneal.copy()
            continue
            
        # Stage 2: Metropolis MC
        c_mc, r_mc = _metropolis_mc(c_anneal, seed_deadline)
        
        if time.time() > seed_deadline:
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc.copy()
            continue
            
        # Stage 3: Smooth polish
        c_polish, r_polish = _smooth_polish(c_mc, seed_deadline, t_start=0.02, t_end=0.001)
        
        if time.time() > seed_deadline:
            if r_polish > best_r:
                best_r = r_polish
                best_c = c_polish.copy()
            continue
            
        # Stage 4: Local search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    final_deadline = deadline - 5.0
    if time.time() < final_deadline:
        # Multi-stage final polish
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Ultra-fine final polish
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.001, t_end=0.00005)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    return best_c
