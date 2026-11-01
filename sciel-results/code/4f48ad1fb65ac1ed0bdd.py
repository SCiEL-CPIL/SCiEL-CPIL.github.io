"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice)
2. Simulated annealing with repulsion to escape local minima
3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
4. Smooth gradient-based polishing (soft-min via log-sum-exp)
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

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Uses only unique pairs (i < j) to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Use only upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]  # shape (N*(N-1)/2,)
    pair_vals = pair_dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall_vals.ravel()
    
    # Concatenate all constraints
    all_vals = np.concatenate([pair_vals, wall_flat])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    # Stable log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    
    weights = exp_vals / s  # normalized weights
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    n_pairs = len(pair_vals)
    pair_w = weights[:n_pairs]  # weights for unique pairs
    
    # For each unique pair (i, j) with i < j:
    # constraint = dist(i,j)/2
    # d(constraint)/d(c_i) = (c_i - c_j) / dist(i,j)  [note: d(dist/2)/d(c_i) = (c_i-c_j)/dist]
    # Wait: d(dist)/d(c_i) = (c_i - c_j)/dist, so d(dist/2)/d(c_i) = (c_i-c_j)/(2*dist)
    # d(constraint)/d(c_j) = (c_j - c_i) / (2*dist)
    
    # grad[i] -= w * (c_i - c_j)/(2*dist)  for the i-index of the pair
    # grad[j] -= w * (c_j - c_i)/(2*dist)  for the j-index of the pair
    
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    
    # Vectorized: for each pair, compute contribution
    # diff_pair[k] = c[pi[k]] - c[pj[k]]
    diff_pair = c[pi] - c[pj]  # shape (n_pairs, 3)
    
    # factor[k] = pair_w[k] / (2 * safe_dists[k])
    factor = pair_w / (2.0 * safe_dists)  # shape (n_pairs,)
    
    # Contribution to grad[pi[k]]: -factor[k] * diff_pair[k]
    # Contribution to grad[pj[k]]: +factor[k] * diff_pair[k]
    
    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_w = weights[n_pairs:]  # N*3 weights
    wall_w = wall_w.reshape(N, 3)
    
    # d(wall_vals[i,k])/d(c[i,k]) = -sign(c[i,k])
    # grad[i,k] -= wall_w[i,k] * (-sign(c[i,k])) = wall_w[i,k] * sign(c[i,k])
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=5000):
    """
    Simulated annealing with repulsion to spread points apart.
    Uses adaptive perturbation size and tracks best radius.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.01
    
    temp = 0.05
    decay = 0.998
    pert_scale = 0.008
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute energy: sum of squared overlaps
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Target: all pair distances >= 2*r_target, wall distances >= r_target
        overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation: larger when stuck, smaller when improving
        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Compute new energy
        diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
        np.fill_diagonal(dists_new, np.inf)
        
        overlap_pair_new = np.maximum(0.0, 2.0 * r_target - dists_new)
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new ** 2) + np.sum(overlap_wall_new ** 2)
        
        # Accept
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                # Reduce perturbation on improvement
                pert_scale *= 0.95
            else:
                # Slightly increase perturbation if no improvement
                pert_scale = min(pert_scale * 1.02, 0.02)
        else:
            # No improvement, increase perturbation
            pert_scale = min(pert_scale * 1.03, 0.02)
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 300 == 0 and E < 1e-4:
            r_target += 0.003
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=200, step_size=0.1):
    """
    Lloyd's algorithm variant: move each center toward the centroid of its
    Voronoi cell (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Add boundary points for Voronoi computation
    # We need points outside the cube to define the Voronoi cells properly
    # Add points at the faces, edges, and corners extended outward
    
    # For Voronoi, we need to include the cube boundaries
    # Add points just outside each face
    boundary_points = []
    eps = 0.1
    
    # Face centers extended outward
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * (0.5 + eps)
            boundary_points.append(p)
    
    # Edge midpoints extended outward
    for i in range(3):
        for j in range(i+1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary_points.append(p)
    
    # Corners extended outward
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)])
                boundary_points.append(p)
    
    boundary_points = np.array(boundary_points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute Voronoi diagram
        all_points = np.vstack([c, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            # Voronoi computation failed, break
            break
        
        # For each original point, find its Voronoi cell and compute centroid
        new_c = c.copy()
        
        for i in range(N):
            # Find the region corresponding to point i
            # vor.regions gives lists of vertex indices
            # vor.point_region[i] gives the region index for point i
            
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            # Get vertices of this region
            verts = vor.vertices[region]
            
            # Clip vertices to the cube [-0.5, 0.5]^3
            verts_clipped = np.clip(verts, -0.5, 0.5)
            
            # Compute centroid of the clipped region
            # Simple centroid: mean of vertices (approximate)
            # Better: use the actual polygon area, but mean is a good approximation
            centroid = np.mean(verts_clipped, axis=0)
            
            # Move toward centroid
            new_c[i] = c[i] + step_size * (centroid - c[i])
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
        
        # Check if this improved the radius
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            # If it got worse, revert
            c = best_c.copy()
        
        # Decrease step size
        step_size *= 0.95
    
    return best_c, best_r

def _epigraph_objective(x, t, lam=1.0):
    """
    Epigraph formulation: minimize -t + lam * sum(max(0, constraint_i - t)^2)
    where constraints are pair distances/2 and wall distances.
    This is differentiable almost everywhere and provides a smooth landscape.
    """
    c = x.reshape(N, 3)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0
    
    # Wall distances
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Violations: max(0, val - t)
    violations = np.maximum(0.0, all_vals - t)
    penalty = np.sum(violations ** 2)
    
    return -t + lam * penalty

def _epigraph_grad(x, t, lam=1.0):
    """
    Gradient of the epigraph objective.
    d(-t)/dt = -1 (but t is fixed here, we only optimize x)
    d(penalty)/d(c) via chain rule through violations.
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0
    
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    violations = np.maximum(0.0, all_vals - t)
    
    # Gradient of penalty w.r.t. all_vals: 2 * violations
    grad_vals = 2.0 * violations
    
    grad = np.zeros_like(c)
    
    n_pairs = len(pair_vals)
    pair_grad = grad_vals[:n_pairs]
    wall_grad = grad_vals[n_pairs:]
    
    # Pair contributions
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    diff_pair = c[pi] - c[pj]
    factor = pair_grad / (2.0 * safe_dists)
    
    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_grad_reshaped = wall_grad.reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_grad_reshaped * signs
    
    return lam * grad.ravel()

def _epigraph_polish(c_init, deadline, t_init=None, lam_init=10.0):
    """
    Polish using epigraph formulation. Start with t at current radius,
    gradually increase t while decreasing lam for smoother convergence.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    if t_init is None:
        t_init = best_r
    
    # Schedule: increase t toward target, decrease lam
    n_steps = 30
    t_target = best_r + 0.015
    t_schedule = np.linspace(t_init, t_target, n_steps)
    lam_schedule = np.logspace(np.log10(lam_init), np.log10(0.1), n_steps)
    
    for si in range(n_steps):
        if time.time() > deadline:
            break
        
        t = t_schedule[si]
        lam = lam_schedule[si]
        
        def obj(x, t=t, lam=lam):
            return _epigraph_objective(x, t, lam)
        
        def grad(x, t=t, lam=lam):
            return _epigraph_grad(x, t, lam)
        
        x0 = c.flatten()
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        max_iter = 80 if si < n_steps // 2 else 120
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                # Update t_target if we improved
                t_target = max(t_target, best_r + 0.01)
                # Reschedule remaining
                remaining = n_steps - si - 1
                if remaining > 0:
                    t_schedule[si+1:] = np.linspace(t, t_target, remaining + 1)[1:]
                    lam_schedule[si+1:] = np.logspace(np.log10(lam), np.log10(0.05), remaining + 1)[1:]
                    
        except Exception:
            continue
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=50):
    """
    Polish the configuration using both softmin and epigraph approaches.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Phase 1: Softmin polish
    n_temps = 15
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        max_iter = 40 + 40 * (ti // (n_temps // 2))
        
        def obj(x, t=t):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x, t=t):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
        
        x0 = c.flatten()
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
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
    
    # Phase 2: Epigraph polish if time remains
    if time.time() < deadline - 10.0:
        c_epi, r_epi = _epigraph_polish(c, deadline, t_init=best_r, lam_init=5.0)
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
    
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
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.002, 0.005, 0.01):
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
            # Pair binding
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.005, 0.01):
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

def _initial_fcc_like():
    """
    Generate an FCC-like initial configuration for 50 points.
    FCC lattice in a cube has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    scaled and shifted to [-0.5, 0.5]^3.
    """
    # FCC basis vectors
    a = 0.5  # lattice constant
    
    # Generate FCC points in a larger region, then select 50
    points = []
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                # FCC has 4 atoms per unit cell
                # Positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                base = np.array([i*a, j*a, k*a])
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0]),
                ]
                for off in offsets:
                    p = base + off
                    # Shift to center around origin
                    p_centered = p - np.array([a, a, a])
                    if np.all(np.abs(p_centered) < 0.6):
                        points.append(p_centered)
    
    points = np.array(points)
    
    # Randomly select 50
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad with random points
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-like
    seeds.append(_initial_fcc_like())
    
    # 2. Random
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 3. Grid-like
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    seeds.append(grid[idx])
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed, giving more time to later seeds
    # since they can benefit from a better initial guess
    total_time = deadline - start_time
    time_per_seed = total_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        # Give later seeds slightly more time
        seed_deadline = seed_start + time_per_seed * (1.0 + 0.1 * i)
        
        # Stage 1: Repulsion annealing (use current best as target)
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=3000)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline, max_iters=100)
        
        # Stage 3: Polish (includes epigraph if time allows)
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.03, t_end=0.0005)
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 15.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 3.0
        
        # Epigraph polish first (smoother landscape)
        c_epi, r_epi = _epigraph_polish(best_c, final_deadline, t_init=best_r, lam_init=3.0)
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
            
        # Softmin polish
        if time.time() < final_deadline - 5.0:
            c_final, r_final = _polish(best_c, final_deadline, t_start=0.005, t_end=0.00005)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
                
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Final ultra-fine epigraph polish
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _epigraph_polish(best_c, final_deadline - 1.0, t_init=best_r, lam_init=1.0)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
