"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice, BCC, shell-core, grid)
2. Simulated annealing with repulsion to escape local minima
3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
4. Smooth gradient-based polishing (soft-min via log-sum-exp)
5. Epigraph formulation for high-precision polish
6. Final local search on constraining pairs/walls
7. Additional: multi-start with diverse random perturbations, active-set refinement
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 50

# EVOLVE-BLOCK-START

# Precompute upper triangle indices for N=50
_IU = np.triu_indices(N, k=1)
_PI, _PJ = _IU
_N_PAIRS = len(_PI)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff * diff, axis=1)
    min_pair = np.sqrt(np.min(dists_sq))
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff_pair = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff_pair * diff_pair, axis=1)
    pair_dists = np.sqrt(dists_sq)
    pair_vals = pair_dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    wall_flat = wall_vals.ravel()
    
    all_vals = np.concatenate([pair_vals, wall_flat])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    
    weights = exp_vals / s
    
    grad = np.zeros_like(c)
    
    pair_w = weights[:_N_PAIRS]
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    factor = pair_w / (2.0 * safe_dists)
    
    np.add.at(grad, _PI, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, _PJ, factor[:, np.newaxis] * diff_pair)
    
    wall_w = weights[_N_PAIRS:]
    wall_w = wall_w.reshape(N, 3)
    
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _epigraph_objective(x, t):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    x = [c.flatten(), t]
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff * diff, axis=1)
    pair_vals = np.sqrt(dists_sq) / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    violations = np.maximum(0, t_val - all_vals)
    penalty = np.sum(violations ** 2)
    
    return -t_val + penalty

def _epigraph_grad(x, t):
    """
    Gradient of epigraph objective.
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff_pair = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff_pair * diff_pair, axis=1)
    pair_dists = np.sqrt(dists_sq)
    pair_vals = pair_dists / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    violations = np.maximum(0, t_val - all_vals)
    
    pair_viol = violations[:_N_PAIRS]
    wall_viol = violations[_N_PAIRS:]
    
    grad = np.zeros_like(x)
    
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    factor = pair_viol / safe_dists
    
    grad_c = grad[:N*3].reshape(N, 3)
    np.add.at(grad_c, _PI, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad_c, _PJ, factor[:, np.newaxis] * diff_pair)
    
    wall_viol_reshaped = wall_viol.reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad[:N*3] += (2.0 * wall_viol_reshaped * signs).reshape(N*3)
    
    grad[N*3] = -1.0 + 2.0 * np.sum(violations)
    
    return grad

def _delaunay_refinement(c_init, deadline, max_iters=100):
    """
    Delaunay-based refinement: use Delaunay edges to identify close pairs
    and apply targeted repulsion along those edges to equalize distances.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            tri = Delaunay(c)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a,b), max(a,b)))
        except Exception:
            break
        
        if not edges:
            break
        
        r_current = _compute_radius(c)
        target_dist = 2.0 * (r_current + 0.0008)
        
        forces = np.zeros_like(c)
        
        for i, j in edges:
            vec = c[i] - c[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            if dist < target_dist:
                deficit = target_dist - dist
                force = (deficit / dist) * vec
                forces[i] += force
                forces[j] -= force
        
        for i in range(N):
            for k in range(3):
                wall_dist = 0.5 - abs(c[i, k])
                if wall_dist < r_current:
                    sign = np.sign(c[i, k])
                    if sign == 0:
                        sign = 1.0
                    forces[i, k] += sign * (r_current - wall_dist)
        
        step = 0.015
        c_new = c + step * forces
        c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            step *= 0.5
            c_new = c + step * forces
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
        
        step *= 0.97
    
    return best_c, best_r

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=8000):
    """
    Simulated annealing with repulsion to spread points apart.
    Adaptive: increases r_target as radius improves, adjusts perturbation scale.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.005
    
    temp = 0.04
    decay = 0.998
    pert_scale = 0.006
    no_improve_count = 0
    patience = 350
    kick_count = 0
    last_r_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        diff = c[_PI] - c[_PJ]
        dists_sq = np.sum(diff * diff, axis=1)
        
        target_dist_sq = (2.0 * r_target) ** 2
        deficit_sq = np.maximum(0.0, target_dist_sq - dists_sq)
        overlap_pair = deficit_sq
        
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation: smaller when close to target
        progress = min(1.0, (best_r - 0.12) / 0.02)
        current_pert = pert_scale * (0.3 + 0.7 * (1.0 - progress)) * (0.5 + temp)
        move = np.random.normal(0.0, current_pert, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        diff_new = c_new[_PI] - c_new[_PJ]
        dists_sq_new = np.sum(diff_new * diff_new, axis=1)
        deficit_sq_new = np.maximum(0.0, target_dist_sq - dists_sq_new)
        overlap_pair_new = deficit_sq_new
        
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new) + np.sum(overlap_wall_new ** 2)
        
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            if it % 3 == 0:
                r_check = _compute_radius(c)
                if r_check > best_r + 1e-10:
                    best_r = r_check
                    best_c = c.copy()
                    no_improve_count = 0
                    last_r_improve = it
                    # Gradually increase target as we improve
                    if E < 1e-3:
                        r_target = max(r_target, best_r + 0.0015)
                else:
                    no_improve_count += 3
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temp *= decay
        
        if no_improve_count > patience:
            # Smarter kick: bias toward increasing minimum distance
            kick_scale = 0.01 + 0.006 * (kick_count % 4)
            kick = np.random.normal(0.0, kick_scale, size=c.shape)
            
            # Add directional bias: push points away from their nearest neighbor
            diff_all = c[_PI] - c[_PJ]
            dists_all = np.sqrt(np.sum(diff_all * diff_all, axis=1))
            min_idx = np.argmin(dists_all)
            i_min, j_min = _PI[min_idx], _PJ[min_idx]
            if dists_all[min_idx] > 1e-8:
                dir_vec = (c[i_min] - c[j_min]) / dists_all[min_idx]
                kick[i_min] += 0.005 * dir_vec
                kick[j_min] -= 0.005 * dir_vec
            
            c_kick = c + kick
            c_kick = np.clip(c_kick, -0.49, 0.49)
            r_kick = _compute_radius(c_kick)
            if r_kick > best_r:
                best_r = r_kick
                best_c = c_kick.copy()
                c = c_kick
            kick_count += 1
            no_improve_count = 0
            temp = max(temp, 0.008)
    
    r_final = _compute_radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=120, step_size=0.15):
    """
    Lloyd's algorithm variant: move each center toward the centroid of its
    Voronoi cell (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    boundary_points = []
    eps = 0.15
    
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * (0.5 + eps)
            boundary_points.append(p)
    
    for i in range(3):
        for j in range(i+1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary_points.append(p)
    
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)])
                boundary_points.append(p)
    
    boundary_points = np.array(boundary_points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        all_points = np.vstack([c, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            break
        
        new_c = c.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            verts = vor.vertices[region]
            verts_clipped = np.clip(verts, -0.5, 0.5)
            centroid = np.mean(verts_clipped, axis=0)
            
            new_c[i] = c[i] + step_size * (centroid - c[i])
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
        
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            c = best_c.copy()
        
        step_size *= 0.95
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.03, t_end=0.00002, iters_per_temp=50):
    """
    Polish the configuration using gradient descent on the softmin radius.
    Uses more temperature steps and tighter tolerances for better convergence.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 30
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        # More iterations for later (colder) temperatures
        progress = ti / n_temps
        max_iter = int(50 + 100 * progress)
        
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
                options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-12:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _epigraph_polish(c_init, deadline, t_offset=0.0005):
    """
    Final high-precision polish using epigraph formulation.
    Multiple restarts with different t_offsets for robustness.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Try multiple t_offsets for better exploration of the feasible region
    offsets = [t_offset, t_offset * 0.5, t_offset * 2.0, t_offset * 0.1]
    
    for off in offsets:
        if time.time() > deadline:
            break
        
        t0 = max(best_r - off, 0.001)
        x0 = np.concatenate([c.flatten(), [t0]])
        
        bounds = [(-0.499999, 0.499999)] * (N * 3) + [(0.0, 0.5)]
        
        try:
            res = minimize(
                _epigraph_objective,
                x0,
                jac=_epigraph_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            x_new = res.x
            c_new = x_new[:N*3].reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r + 1e-12:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
    
    return best_c, best_r

def _active_set_polish(c_init, deadline, max_iters=200):
    """
    Active-set refinement: identify the constraining pairs and walls,
    then solve a small QP-like problem to push them to equal clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        r = _compute_radius(c)
        
        # Find constraining pairs (distance <= 2*r + epsilon)
        diff = c[_PI] - c[_PJ]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        constraining_pairs = np.where(dists <= 2.0 * r + 0.001)[0]
        
        # Find constraining walls
        wall_dists = 0.5 - np.abs(c)
        constraining_walls = np.where(wall_dists <= r + 0.001)
        
        # Build force from constraining constraints
        forces = np.zeros_like(c)
        
        for idx in constraining_pairs:
            i, j = _PI[idx], _PJ[idx]
            vec = c[i] - c[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            # Push apart
            push = (2.0 * r + 0.0005 - dist) / dist * vec
            forces[i] += push
            forces[j] -= push
        
        for i, k in zip(*constraining_walls):
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            forces[i, k] += sign * (r + 0.0005 - wall_dists[i, k])
        
        if np.all(forces == 0):
            break
        
        # Line search
        step = 0.01
        improved = False
        for _ in range(10):
            c_try = c + step * forces
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try a small random perturbation
            c_try = c + np.random.normal(0, 0.002, c.shape)
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=300):
    """
    Local search: Perturb points that are constraining the radius.
    Includes shake mechanism for escaping local minima.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
            c = best_c.copy()
            r = best_r
             
        diff = c[_PI] - c[_PJ]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        min_pair_dist = np.min(dists)
        
        improved = False
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        else:
            min_idx = np.argmin(dists)
            i, j = _PI[min_idx], _PJ[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            # Shake mechanism: perturb multiple nearby points
            if no_improve > 15 and it % 3 == 0:
                c_try = c.copy()
                # Find the 3 closest pairs and perturb them
                sorted_dists = np.argsort(dists)
                for idx in sorted_dists[:3]:
                    ii, jj = _PI[idx], _PJ[idx]
                    rand_move = np.random.normal(0, 0.004, size=(2, 3))
                    c_try[ii] += rand_move[0]
                    c_try[jj] += rand_move[1]
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    no_improve = 0
    
    return best_c, best_r

def _initial_fcc_like():
    """
    Generate an FCC-like initial configuration for 50 points.
    """
    a = 0.5
    points = []
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                base = np.array([i*a, j*a, k*a])
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0]),
                ]
                for off in offsets:
                    p = base + off
                    p_centered = p - np.array([a, a, a])
                    if np.all(np.abs(p_centered) < 0.55):
                        points.append(p_centered)
    
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_bcc():
    """
    Generate a BCC-like initial configuration.
    """
    a = 0.28
    points = []
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p = np.array([i, j, k]) * a
                p_centered = p - np.array([0.5*a, 0.5*a, 0.5*a])
                if np.all(np.abs(p_centered) < 0.48):
                    points.append(p_centered)
                # BCC body center
                p_bcc = (np.array([i, j, k]) + 0.5) * a
                p_bcc_centered = p_bcc - np.array([0.5*a, 0.5*a, 0.5*a])
                if np.all(np.abs(p_bcc_centered) < 0.48):
                    points.append(p_bcc_centered)
    
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    points = []
    
    # Core: BCC-like lattice
    a_core = 0.2
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                p = np.array([i, j, k]) * a_core
                if np.linalg.norm(p) < 0.32:
                    points.append(p)
                p_bcc = (np.array([i, j, k]) + 0.5) * a_core
                if np.linalg.norm(p_bcc) < 0.32:
                    points.append(p_bcc)
    
    # Shell: points near faces
    a_shell = 0.36
    face_positions = []
    for axis in range(3):
        for sign in [-1, 1]:
            for u in np.linspace(-0.28, 0.28, 3):
                for v in np.linspace(-0.28, 0.28, 3):
                    p = np.zeros(3)
                    p[axis] = sign * a_shell
                    other_axes = [ax for ax in range(3) if ax != axis]
                    p[other_axes[0]] = u
                    p[other_axes[1]] = v
                    if np.all(np.abs(p) < 0.48):
                        face_positions.append(p)
    
    points.extend(face_positions)
    
    # Edge and corner points
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p_corner = np.array([sx, sy, sz]) * 0.4
                if np.all(np.abs(p_corner) < 0.48):
                    points.append(p_corner)
    
    points = np.array(points)
    
    if len(points) > 0:
        unique = [points[0]]
        for p in points[1:]:
            if not any(np.linalg.norm(p - u) < 0.05 for u in unique):
                unique.append(p)
        points = np.array(unique)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_grid():
    """
    Generate a structured grid initial configuration.
    """
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _initial_random_perturbed(base_c, scale=0.05):
    """
    Generate a perturbed version of an existing configuration.
    """
    c = base_c.copy()
    c += np.random.normal(0, scale, c.shape)
    c = np.clip(c, -0.45, 0.45)
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    seeds = []
    seeds.append(_initial_fcc_like())
    seeds.append(_initial_bcc())
    seeds.append(_initial_shell_core())
    seeds.append(_initial_grid())
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Focused time allocation: 3 main seeds get substantial time,
    # remaining time goes to final polishing
    time_allocs = [40.0, 35.0, 30.0, 15.0, 12.0]
    
    for i, s in enumerate(seeds):
        remaining = deadline - time.time()
        if remaining < 25.0:
            break
        
        alloc = min(time_allocs[i] if i < len(time_allocs) else 8.0, remaining - 25.0)
        if alloc < 12.0:
            break
        
        seed_deadline = time.time() + alloc
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=max(best_r + 0.002, 0.125), max_iters=8000)
        
        if time.time() > seed_deadline - 8.0:
            continue
        
        # Stage 2: Delaunay refinement
        c_delaunay, r_delaunay = _delaunay_refinement(c_anneal, seed_deadline, max_iters=80)
        
        if time.time() > seed_deadline - 8.0:
            continue
        
        # Stage 3: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_delaunay, seed_deadline, max_iters=80)
        
        if time.time() > seed_deadline - 8.0:
            continue
        
        # Stage 4: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.03, t_end=0.00002)
        
        if time.time() > seed_deadline - 5.0:
            continue
        
        # Stage 5: Active-set polish
        c_as, r_as = _active_set_polish(c_polish, seed_deadline, max_iters=100)
        
        if time.time() > seed_deadline - 3.0:
            continue
        
        # Stage 6: Local Search
        c_ls, r_ls = _local_search(c_as, seed_deadline, max_iters=200)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Additional multi-start: perturb the best solution and re-optimize
    remaining = deadline - time.time()
    if remaining > 30.0:
        for pert_scale in [0.03, 0.05, 0.08]:
            if time.time() > deadline - 25.0:
                break
            
            c_pert = _initial_random_perturbed(best_c, scale=pert_scale)
            pert_deadline = time.time() + min(20.0, deadline - time.time() - 25.0)
            
            c_a, r_a = _repulsion_anneal(c_pert, pert_deadline, r_target=best_r + 0.001, max_iters=4000)
            if time.time() > pert_deadline - 5.0:
                continue
            c_p, r_p = _polish(c_a, pert_deadline, t_start=0.02, t_end=0.00005)
            if time.time() > pert_deadline - 3.0:
                continue
            c_l, r_l = _local_search(c_p, pert_deadline, max_iters=100)
            
            if r_l > best_r:
                best_r = r_l
                best_c = c_l.copy()
    
    # Final high-precision polish on the global best
    remaining = deadline - time.time()
    if remaining > 15.0:
        final_deadline = deadline - 2.0
        
        # Epigraph polish with multiple offsets
        c_epi, r_epi = _epigraph_polish(best_c, min(final_deadline, time.time() + 20.0), t_offset=0.001)
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
        
        remaining = deadline - time.time()
        if remaining > 10.0:
            c_final, r_final = _polish(best_c, final_deadline, t_start=0.005, t_end=0.00001)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
            
        remaining = deadline - time.time()
        if remaining > 8.0:
            c_epi2, r_epi2 = _epigraph_polish(best_c, final_deadline, t_offset=0.0005)
            if r_epi2 > best_r:
                best_r = r_epi2
                best_c = c_epi2.copy()
            
        remaining = deadline - time.time()
        if remaining > 6.0:
            c_as, r_as = _active_set_polish(best_c, final_deadline, max_iters=150)
            if r_as > best_r:
                best_r = r_as
                best_c = c_as.copy()
        
        remaining = deadline - time.time()
        if remaining > 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        remaining = deadline - time.time()
        if remaining > 3.0:
            c_final2, r_final2 = _polish(best_c, final_deadline, t_start=0.001, t_end=0.000001)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
        
        remaining = deadline - time.time()
        if remaining > 2.0:
            c_epi3, r_epi3 = _epigraph_polish(best_c, final_deadline, t_offset=0.0002)
            if r_epi3 > best_r:
                best_r = r_epi3
                best_c = c_epi3.copy()
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
