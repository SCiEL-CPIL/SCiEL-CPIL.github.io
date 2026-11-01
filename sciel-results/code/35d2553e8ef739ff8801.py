"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice)
2. Vectorized repulsion-based Metropolis Monte Carlo for global exploration
3. Epigraph-based smooth optimization for precise radius maximization
4. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
5. Final high-precision polish pass
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
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _repulsion_energy(c, r_target):
    """
    Vectorized repulsion energy: sum of squared overlaps with target radius.
    Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i sum_k max(0, r - w_ik)^2
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
    energy = np.sum(overlap_pair ** 2)
    
    wall_dists = 0.5 - np.abs(c)
    overlap_wall = np.maximum(0.0, r_target - wall_dists)
    energy += np.sum(overlap_wall ** 2)
    
    return energy

def _metropolis_mc(c_init, deadline, r_start=0.10, r_step=0.0002, max_iters=5000):
    """
    Metropolis Monte Carlo with vectorized repulsion energy.
    Explores global landscape by accepting worse states with temperature.
    """
    c = c_init.copy()
    current_r = r_start
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.03
    decay = 0.9995
    step_scale = 0.015
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        E = _repulsion_energy(c, current_r)
        
        # Increase radius if feasible
        if E < 1e-6:
            current_r += r_step
            step_scale = max(0.003, step_scale * 0.998)
        else:
            temp *= decay
            step_scale = max(0.002, step_scale * 0.999)
        
        # Random perturbation
        move = np.random.normal(0.0, step_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        E_new = _repulsion_energy(c_new, current_r)
        
        # Metropolis acceptance
        dE = E_new - E
        if dE < 0 or (temp > 1e-8 and np.random.random() < np.exp(-dE / temp)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
        
        # Periodic Lloyd step
        if it % 500 == 250:
            c_lloyd = _lloyd_step(c, deadline, n_iter=5)
            r_lloyd = _compute_radius(c_lloyd)
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_c = c_lloyd.copy()
                c = c_lloyd.copy()
                
    return best_c, best_r

def _epigraph_obj_and_grad(x, r_target):
    """
    Epigraph formulation: minimize sum of soft violations.
    For each constraint v_j >= r_target, we penalize max(0, r_target - v_j)^2.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Violations: how much each constraint is violated
    pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dists)
    wall_dists = 0.5 - np.abs(c)
    wall_viol = np.maximum(0.0, r_target - wall_dists)
    
    # Objective: sum of squared violations
    obj = np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Pair gradient: d/d(c_i) max(0, 2r - |c_i - c_j|)^2
    # = -2 * max(0, 2r - d) * (c_i - c_j) / d  for violated pairs
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        v = pair_viol[idx]
        if v > 1e-12:
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                g = -2.0 * v * d / dist
                grad[i] += g
                grad[j] -= g
    
    # Wall gradient: d/d(c_ik) max(0, r - (0.5 - |c_ik|))^2
    # = -2 * max(0, r - w) * sign(c_ik)
    wall_viol_reshaped = wall_viol.reshape(N, 3)
    signs = np.sign(c)
    grad += 2.0 * wall_viol_reshaped * signs
    
    return obj, grad

def _epigraph_polish(c_init, deadline, r_target=None, n_iters=10):
    """
    Polish using epigraph formulation with L-BFGS-B.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    if r_target is None:
        r_target = best_r + 0.001
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        def obj_and_grad(x):
            return _epigraph_obj_and_grad(x, r_target)
        
        x0 = c.flatten()
        bounds = [(-0.49999, 0.49999)] * (N * 3)
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                # Try to increase target
                r_target = min(0.15, r_new + 0.0005)
            else:
                # Decrease target if stuck
                r_target = max(0.08, r_target - 0.0002)
                
        except Exception:
            continue
            
    return best_c, best_r

def _softmin_radius_and_grad(c, t):
    """
    Smooth approximation of the radius with analytical gradient (vectorized).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    all_vals = np.concatenate([pair_dists, wall_vals.ravel()])
    
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    value = m - t * np.log(s)
    
    grad = np.zeros_like(c)
    
    # Vectorized pair gradient
    weights_pair = exp_vals[:len(pair_dists)] / s
    
    # For each pair (i,j), contribution to grad[i] += w * (c_i - c_j) / |c_i - c_j|
    # and grad[j] -= w * (c_i - c_j) / |c_i - c_j|
    iu_i, iu_j = iu[0], iu[1]
    pair_diffs = c[iu_i] - c[iu_j]
    pair_norms = np.linalg.norm(pair_diffs, axis=1)
    pair_norms_safe = np.where(pair_norms > 1e-12, pair_norms, 1.0)
    pair_dirs = pair_diffs / pair_norms_safe[:, np.newaxis]
    
    # Accumulate using scatter
    for idx in range(len(iu_i)):
        w = weights_pair[idx]
        if w < 1e-10:
            continue
        grad[iu_i[idx]] += w * pair_dirs[idx]
        grad[iu_j[idx]] -= w * pair_dirs[idx]
    
    # Wall contributions
    weights_wall = exp_vals[len(pair_dists):] / s
    wall_weights = weights_wall.reshape(N, 3)
    signs = np.sign(c)
    grad -= wall_weights * signs
    
    return value, grad

def _lloyd_step(c, deadline, n_iter=50):
    """
    Lloyd's algorithm: move each sphere to the centroid of its Voronoi cell.
    """
    c = c.copy()
    for it in range(n_iter):
        if time.time() > deadline:
            break
        try:
            vor = Voronoi(c)
            new_c = c.copy()
            for i in range(N):
                region = vor.regions[vor.point_region[i]]
                if -1 in region or len(region) == 0:
                    continue
                vertices = vor.vertices[region]
                if len(vertices) == 0:
                    continue
                centroid = np.mean(vertices, axis=0)
                new_c[i] = np.clip(centroid, -0.45, 0.45)
            c = new_c
        except Exception:
            break
    return c

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
            for step in (0.003, 0.006, 0.01, 0.02):
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
            
            for step in (0.003, 0.006, 0.01, 0.02):
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

def _initial_fcc_like(n_points):
    """
    Generate an FCC-like initial configuration with jitter.
    """
    pts = []
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                base = np.array([ix, iy, iz]) * 0.5
                for dx in [0, 0.25]:
                    for dy in [0, 0.25]:
                        for dz in [0, 0.25]:
                            p = base + np.array([dx, dy, dz])
                            if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                                pts.append(p)
                for face in range(3):
                    for val in [0.125, 0.375]:
                        p = base + np.zeros(3)
                        p[face] = val
                        if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                            pts.append(p)
    
    pts = np.array(pts)
    pts += np.random.normal(0, 0.02, size=pts.shape)
    pts = np.clip(pts, -0.45, 0.45)
    
    if len(pts) > n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    else:
        while len(pts) < n_points:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, size=(1, 3))])
    
    return pts[:n_points]

def _initial_bcc_like(n_points):
    """
    Generate a BCC-like initial configuration with jitter.
    BCC: corners + body center
    """
    pts = []
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                base = np.array([ix, iy, iz]) * 0.5
                # Corners
                for dx in [0, 0.25]:
                    for dy in [0, 0.25]:
                        for dz in [0, 0.25]:
                            p = base + np.array([dx, dy, dz])
                            if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                                pts.append(p)
                # Body center
                p = base + np.array([0.125, 0.125, 0.125])
                if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                    pts.append(p)
    
    pts = np.array(pts)
    pts += np.random.normal(0, 0.02, size=pts.shape)
    pts = np.clip(pts, -0.45, 0.45)
    
    if len(pts) > n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    else:
        while len(pts) < n_points:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, size=(1, 3))])
    
    return pts[:n_points]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-like structured
    seeds.append(_initial_fcc_like(N))
    
    # 2. BCC-like structured
    seeds.append(_initial_bcc_like(N))
    
    # 3. Jittered simple cubic
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    grid_pts = grid[idx] + np.random.normal(0, 0.01, size=(N, 3))
    seeds.append(np.clip(grid_pts, -0.45, 0.45))
    
    # 4. Random
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: 60% to MC exploration, 25% to epigraph polish, 15% to local search
    total_time = deadline - start_time
    mc_time = 0.60 * total_time
    epigraph_time = 0.25 * total_time
    local_time = 0.15 * total_time
    
    time_per_seed_mc = mc_time / len(seeds)
    
    # Phase 1: Metropolis MC for each seed
    for i, s in enumerate(seeds):
        seed_deadline = start_time + (i + 1) * time_per_seed_mc
        if time.time() > seed_deadline - 1.0:
            break
        
        c_mc, r_mc = _metropolis_mc(s, seed_deadline, r_start=0.10, max_iters=4000)
        
        if r_mc > best_r:
            best_r = r_mc
            best_c = c_mc.copy()
    
    # Phase 2: Epigraph polish on best configuration
    if time.time() < deadline - 10.0:
        epigraph_deadline = deadline - 10.0
        c_epi, r_epi = _epigraph_polish(best_c, epigraph_deadline, r_target=best_r + 0.001, n_iters=8)
        
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
    
    # Phase 3: Lloyd + local search
    if time.time() < deadline - 5.0:
        local_deadline = deadline - 3.0
        c_lloyd = _lloyd_step(best_c, local_deadline, n_iter=15)
        r_lloyd = _compute_radius(c_lloyd)
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
        
        c_ls, r_ls = _local_search(best_c, local_deadline, max_iters=600)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Final quick polish if time remains
    if time.time() < deadline - 2.0:
        c_final, r_final = _epigraph_polish(best_c, deadline - 1.0, r_target=best_r + 0.0005, n_iters=3)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    return best_c

# EVOLVE-BLOCK-END
