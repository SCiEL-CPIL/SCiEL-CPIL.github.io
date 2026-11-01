"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Physics-based Simulated Annealing to find a good packing for a 
target radius, followed by gradient-based polishing to maximize the radius.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    Uses log-sum-exp for numerical stability.
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for stability
    pair_log_weights = -pair_vals / t
    np.fill_diagonal(pair_log_weights, -np.inf)
    
    wall_log_weights = -wall_vals / t
    
    # Combine: softmin = -t * log(sum(exp(log_weights)))
    # For numerical stability, subtract max
    all_log_weights = np.concatenate([pair_log_weights.ravel(), wall_log_weights.ravel()])
    max_lw = np.max(all_log_weights)
    if np.isinf(max_lw):
        m = np.min(np.concatenate([pair_vals.ravel(), wall_vals.ravel()]))
        return m, np.zeros_like(c)
    
    shifted = all_log_weights - max_lw
    log_sum = max_lw + np.log(np.sum(np.exp(shifted)))
    softmin_val = -t * log_sum
    
    # Gradient: d(softmin)/d(c_ik) = -t * d(log_sum)/d(c_ik)
    # d(log_sum)/d(c_ik) = sum_j exp(log_w_j) * d(log_w_j)/d(c_ik) / sum_j exp(log_w_j)
    # = sum_j w_j * d(log_w_j)/d(c_ik) where w_j = exp(log_w_j - log_sum)
    
    grad_c = np.zeros_like(c)
    
    # Pair contributions: log_w_ij = -d_ij/(2t), d(log_w_ij)/d(c_i) = -unit_ij/(2t)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Weights for pairs (normalized)
    pair_exp = np.exp(pair_log_weights - max_lw)
    np.fill_diagonal(pair_exp, 0.0)
    
    total_exp = np.sum(pair_exp) + np.sum(np.exp(wall_log_weights - max_lw))
    
    if total_exp < 1e-300:
        return softmin_val, np.zeros_like(c)
    
    # Pair gradient contribution
    grad_pair = -0.5 / t * pair_exp[:, :, np.newaxis] * unit_vecs
    grad_c += np.sum(grad_pair, axis=1)
    
    # Wall gradient: log_w_ik = -(0.5 - |c_ik|)/t, d/d(c_ik) = -sign(c_ik)/t
    wall_exp = np.exp(wall_log_weights - max_lw)
    signs = np.sign(c)
    grad_wall = -1.0 / t * wall_exp * signs
    grad_c += grad_wall
    
    # Normalize by total_exp
    grad_c /= total_exp
    
    return softmin_val, grad_c

def _anneal_objective(x, r_target):
    """
    Objective for simulated annealing phase.
    We want to minimize the "overlap" energy if the target radius is not satisfied.
    Energy = sum over pairs max(0, r_target - d_ij)^2 + sum over walls max(0, r_target - w_i)^2
    If we can achieve r_target with zero energy, we can increase r_target.
    """
    c = x.reshape(N, 3)
    energy = 0.0
    
    # Pairwise overlaps
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Overlap amount for each pair
    # If dist < 2*r_target, overlap is (2*r_target - dist)
    # Squared overlap to make it smooth
    overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
    energy += np.sum(overlap_pair ** 2)
    
    # Wall overlaps
    wall_dists = 0.5 - np.abs(c)
    overlap_wall = np.maximum(0.0, r_target - wall_dists)
    energy += np.sum(overlap_wall ** 2)
    
    return energy

def _simulate_anneal(c_init, deadline, r_init=0.05, r_step=0.0005, max_iters=2000):
    """
    Perform simulated annealing to find a configuration that supports a growing radius.
    """
    c = c_init.copy()
    current_r = r_init
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.1
    decay = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current energy
        E = _anneal_objective(c.flatten(), current_r)
        
        # Try to increase radius
        if E < 1e-6:
            # We have room to increase radius
            current_r += r_step
            # Reset temperature slightly to escape potential local minima in position
            if it % 100 == 0:
                temp = max(0.01, temp * 1.1)
        else:
            # Decrease temperature
            temp *= decay
        
        # Perturbation
        # Random move
        move = np.random.normal(0.0, 0.01, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        E_new = _anneal_objective(c_new.flatten(), current_r)
        
        # Accept if better or with probability exp(-dE/T)
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / temp):
            c = c_new
            # Check if this configuration is actually valid for a larger radius
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.001, n_temps=6):
    """
    Polish the configuration using gradient descent on the softmin radius with analytical gradients.
    Fewer temperature steps, tighter convergence for speed.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
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
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _lloyd_step(c, iterations=100, clip_margin=0.001):
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
            # Get the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - keep current position or move toward center
                # For unbounded regions, we can't easily compute centroid
                # Instead, nudge slightly toward the cube center
                center = np.array([0.0, 0.0, 0.0])
                direction = center - c[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                new_c[i] = c[i] + 0.01 * direction
                continue
            
            # Get the vertices of this Voronoi cell
            vertices = vor.vertices[region]
            
            if len(vertices) < 3:
                # Degenerate case, keep current position
                continue
            
            # Compute centroid of the Voronoi cell
            centroid = np.mean(vertices, axis=0)
            
            # Clip to cube boundary with a small margin
            centroid = np.clip(centroid, -0.5 + clip_margin, 0.5 - clip_margin)
            
            # Move toward centroid with a damping factor
            new_c[i] = 0.7 * c[i] + 0.3 * centroid
        
        c = new_c
    
    return c

def _lloyd_optimize(c_init, deadline, max_rounds=50):
    """
    Run Lloyd's algorithm with periodic radius checks and restarts.
    Returns the best configuration found.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Run a batch of Lloyd iterations
        c = _lloyd_step(c, iterations=20)
        
        # Check improvement
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        else:
            # If no improvement, try a small random perturbation to escape local optima
            if round_idx % 5 == 4:
                c = best_c.copy()
                perturb_idx = np.random.choice(N, size=5, replace=False)
                for idx in perturb_idx:
                    c[idx] += np.random.normal(0, 0.02, size=3)
                c = np.clip(c, -0.49, 0.49)
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    Tries multiple nearby constraint pairs and random perturbations.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                # Reset to best and add random perturbation
                c = best_c.copy()
                n_perturb = np.random.randint(2, 6)
                idxs = np.random.choice(N, size=n_perturb, replace=False)
                for idx in idxs:
                    c[idx] += np.random.normal(0, 0.015, size=3)
                c = np.clip(c, -0.499999, 0.499999)
                no_improve = 0
                continue
        
        # Find top-5 binding constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.abs(c)
        
        # Get the 5 smallest pair distances
        pair_flat = dists.ravel()
        upper_tri_idx = np.triu_indices(N, k=1)
        pair_dists = dists[upper_tri_idx]
        
        n_top = min(5, len(pair_dists))
        top_pair_indices = np.argsort(pair_dists)[:n_top]
        
        # Get the 5 smallest wall distances
        wall_flat = wall_dists.ravel()
        top_wall_indices = np.argsort(wall_flat)[:min(5, len(wall_flat))]
        
        moved = False
        
        # Try moving apart top pair constraints
        for pi in top_pair_indices:
            i, j = upper_tri_idx[0][pi], upper_tri_idx[1][pi]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.003, 0.005, 0.008, 0.012):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    moved = True
                    break
            if moved:
                break
        
        # Try moving away from wall constraints
        if not moved:
            for wi in top_wall_indices:
                i, k = np.unravel_index(wi, wall_dists.shape)
                sign = 1.0 if c[i, k] > 0 else -1.0
                for step in (0.003, 0.005, 0.008, 0.012):
                    c_try = c.copy()
                    c_try[i, k] += step * sign
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
                        moved = True
                        break
                if moved:
                    break
        
        # If nothing moved, try a random small perturbation
        if not moved:
            idx = np.random.randint(N)
            c[idx] += np.random.normal(0, 0.008, size=3)
            c = np.clip(c, -0.499999, 0.499999)
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations - 3 diverse seeds
    seeds = []
    
    # 1. Random uniform
    seeds.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 2. Grid-like with jitter
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    grid_seed = grid[idx].copy()
    grid_seed += np.random.normal(0, 0.02, size=grid_seed.shape)
    grid_seed = np.clip(grid_seed, -0.49, 0.49)
    seeds.append(grid_seed)
    
    # 3. FCC-inspired structured start
    # Use a 3x4x5 grid (60 points) and pick 50
    nx, ny, nz = 3, 4, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid3 = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    idx3 = np.random.choice(len(grid3), N, replace=False)
    fcc_seed = grid3[idx3].copy()
    fcc_seed += np.random.normal(0, 0.015, size=fcc_seed.shape)
    fcc_seed = np.clip(fcc_seed, -0.49, 0.49)
    seeds.append(fcc_seed)

    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: 3 seeds, each gets roughly 1/3 of time minus final refinement
    n_seeds = len(seeds)
    final_reserve = 25.0  # seconds reserved for final refinement
    time_per_seed = (deadline - start_time - final_reserve) / n_seeds
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        # Stage 1: Annealing (larger share of time)
        anneal_deadline = seed_start + 0.4 * time_per_seed
        c_anneal, r_anneal = _simulate_anneal(s, anneal_deadline, max_iters=1500)
        
        # Stage 2: Lloyd's algorithm
        lloyd_deadline = seed_start + 0.7 * time_per_seed
        if time.time() < lloyd_deadline:
            c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, lloyd_deadline, max_rounds=25)
        else:
            c_lloyd, r_lloyd = c_anneal, r_anneal
        
        # Stage 3: Polish
        polish_deadline = seed_start + 0.9 * time_per_seed
        if time.time() < polish_deadline:
            c_polish, r_polish = _polish(c_lloyd, polish_deadline, t_start=0.08, t_end=0.005, n_temps=5)
        else:
            c_polish, r_polish = c_lloyd, r_lloyd
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - final_reserve - 2.0:
            break
            
    # Final refinement on the global best
    final_start = time.time()
    final_deadline = deadline - 3.0
    
    # Final Lloyd's pass
    if time.time() < final_deadline - 15.0:
        c_lloyd_final, r_lloyd_final = _lloyd_optimize(best_c, final_deadline - 12.0, max_rounds=15)
        if r_lloyd_final > best_r:
            best_r = r_lloyd_final
            best_c = c_lloyd_final.copy()
    
    # Final high-precision polish
    if time.time() < final_deadline - 5.0:
        c_final, r_final = _polish(best_c, final_deadline - 3.0, t_start=0.03, t_end=0.002, n_temps=4)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # One last local search with many iterations
    if time.time() < final_deadline - 1.0:
        c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=800)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    return best_c

# EVOLVE-BLOCK-END
