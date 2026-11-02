"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initial configurations (FCC-inspired, grid, random, BCC)
2. Fast repulsion-based simulated annealing for global search
3. Softmin gradient polishing with adaptive temperature
4. Active-set epigraph polishing for final precision
5. Local search on binding constraints
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

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

def _seed_bcc():
    """
    Generate a BCC-inspired initial configuration for 50 spheres.
    BCC has 2 atoms per unit cell (corners + center).
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
                # BCC: center of cell
                points.append([x, y, z])
                # Corners (but only keep those inside)
                for dx, dy, dz in [(0,0,0),(0,0,spacing),(0,spacing,0),(0,spacing,spacing),
                                    (spacing,0,0),(spacing,0,spacing),(spacing,spacing,0),(spacing,spacing,spacing)]:
                    px = x + dx - spacing/2
                    py = y + dy - spacing/2
                    pz = z + dz - spacing/2
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

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    
    # Softmin = -t * log(sum(exp(-val/t)))
    # For stability: let m = min(all_vals), then softmin = m - t*log(sum(exp(-(val-m)/t)))
    m = np.min(all_vals)
    shifted = all_vals - m
    weights = np.exp(-shifted / t)
    total_sum = np.sum(weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
        
    softmin_val = m - t * np.log(total_sum)
    
    # Gradient
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    pair_weights = np.exp(-(pair_vals - m) / t)
    np.fill_diagonal(pair_weights, 0.0)
    
    grad_pair_contrib = -0.5 * pair_weights[:, :, np.newaxis] * unit_vecs
    grad_c += np.sum(grad_pair_contrib, axis=1)
    
    # Wall contributions
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    wall_weights = np.exp(-(wall_vals - m) / t)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=40):
    """
    Repulsion-based simulated annealing.
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
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, 1e10)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff / dists[:, :, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, :, np.newaxis] * unit_vecs
            forces = np.sum(pair_forces, axis=1)
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists * wall_dists)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
            
            signs = np.sign(c)
            signs[signs == 0] = 1.0
            forces += wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.1, size=c.shape)
            c_new = c + forces * 0.01 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(delta_r / max(t, 1e-10)):
                c = c_new
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=100):
    """
    Polish using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 12)
    
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
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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

def _active_set_polish(c_init, deadline, max_iters=100):
    """
    Active-set polishing: identify binding constraints and optimize only those.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        r_cur = _compute_radius(c)
        
        # Find binding constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        # If no improvement possible, break
        if r_cur <= best_r * 0.9999:
            break
        
        # Define objective with penalty for constraint violations
        def obj(x):
            centers = x.reshape(N, 3)
            
            # Pair constraints
            diff_c = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_c = np.sqrt(np.sum(diff_c * diff_c, axis=-1))
            np.fill_diagonal(dists_c, np.inf)
            pair_viol = np.maximum(0, 2 * r_cur - dists_c)
            
            # Wall constraints
            wall_dists_c = 0.5 - np.abs(centers)
            wall_viol = np.maximum(0, r_cur - wall_dists_c)
            
            penalty = 0.5 * (np.sum(pair_viol**2) + np.sum(wall_viol**2))
            return -r_cur + 100.0 * penalty
        
        def grad(x):
            centers = x.reshape(N, 3)
            
            diff_c = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_c = np.sqrt(np.sum(diff_c * diff_c, axis=-1))
            np.fill_diagonal(dists_c, np.inf)
            
            viol = np.maximum(0, 2 * r_cur - dists_c)
            viol[dists_c < 1e-10] = 0
            
            with np.errstate(divide='ignore', invalid='ignore'):
                unit_vecs = diff_c / dists_c[:, :, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            factor = 100.0 * viol
            grad_centers = -np.sum(factor[:, :, np.newaxis] * unit_vecs, axis=1)
            
            wall_dists_c = 0.5 - np.abs(centers)
            wall_viol = np.maximum(0, r_cur - wall_dists_c)
            signs = np.sign(centers)
            signs[signs == 0] = 1.0
            grad_centers += 100.0 * wall_viol * signs
            
            return grad_centers.ravel()
        
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
            else:
                # No improvement, stop
                break
                
        except Exception:
            continue
    
    return best_c, best_r

def _metropolis_mc(c_init, deadline, n_iters=5000, temp=0.01):
    """
    Metropolis Monte Carlo.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    cur_r = best_r
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(N)
        delta = np.random.normal(0, temp * 0.1, size=3)
        c_new = c.copy()
        c_new[idx] += delta
        c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        delta_r = r_new - cur_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(temp, 1e-10)):
            c = c_new
            cur_r = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= 0.999
    
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
        if r <= best_r:
            c = best_c.copy()
            r = best_r
             
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def _nelder_mead_polish(c_init, deadline, maxiter=200):
    """
    Final high-precision polish using Nelder-Mead.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def obj(x):
        centers = x.reshape(N, 3)
        centers = np.clip(centers, -0.499999, 0.499999)
        r = _compute_radius(centers)
        return -r
    
    x0 = c.flatten()
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        c_new = res.x.reshape(N, 3)
        c_new = np.clip(c_new, -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
    except Exception:
        pass
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_bcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())

    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time) / n_seeds
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
        
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - 5.0)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, t_start=2.0, t_end=0.05, iters_per_temp=30)
        
        # Stage 2: Metropolis MC
        if time.time() < seed_deadline - 5.0:
            c_mc, r_mc = _metropolis_mc(c_anneal, seed_deadline, n_iters=3000, temp=0.01)
            if r_mc > r_anneal:
                c_anneal = c_mc
                r_anneal = r_mc
        
        # Stage 3: Smooth polish
        c_polish, r_polish = _smooth_polish(c_anneal, seed_deadline, t_start=0.05, t_end=0.005)
        
        # Stage 4: Active-set polish
        c_active, r_active = _active_set_polish(c_polish, seed_deadline, max_iters=100)
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_active, seed_deadline, max_iters=100)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 3.0
        
        # Smooth polish again
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.03, t_end=0.0001, iters_per_temp=80)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Active-set polish
        if time.time() < final_deadline - 5.0:
            c_active_final, r_active_final = _active_set_polish(best_c, final_deadline, max_iters=150)
            if r_active_final > best_r:
                best_r = r_active_final
                best_c = c_active_final.copy()
        
        # Nelder-Mead polish
        if time.time() < final_deadline - 3.0:
            c_nm, r_nm = _nelder_mead_polish(best_c, final_deadline, maxiter=100)
            if r_nm > best_r:
                best_r = r_nm
                best_c = c_nm.copy()
        
        # One last local search
        if time.time() < final_deadline - 1.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
