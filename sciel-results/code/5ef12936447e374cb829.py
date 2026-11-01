import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _seed_fcc():
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
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
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_bcc():
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
    if len(c) == 0:
        return 0.0
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    wall_vals = 0.5 - np.abs(c)
    
    # Softmin
    # To avoid overflow/underflow, we can use log-sum-exp trick if needed, 
    # but for t > 0.01 and values in [0, 0.5], exp is safe.
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return 0.0, np.zeros_like(c)
        
    softmin_val = -t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Gradient of softmin: 
    # d/dx_i (-t * log(sum exp(-f_j/t))) = sum (exp(-f_j/t) / sum) * (1/t) * (-t) * df_j/dx_i
    # = sum (w_j) * (-df_j/dx_i)
    # f_pair = dist/2. df/dx_i = 0.5 * (x_i - x_j)/dist
    # f_wall = 0.5 - |x|. df/dx_i = -sign(x_i)
    
    # Pair contribution to grad:
    # -0.5 * w * (x_i - x_j)/dist
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_c, _iu, -pair_grad)
    np.add.at(grad_c, _ju, pair_grad)
    
    # Wall contribution to grad:
    # -(-sign(x)) * w = sign(x) * w
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
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
            
            forces = np.zeros_like(c)
            
            diff = c[_iu] - c[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            np.add.at(forces, _iu, pair_forces)
            np.add.at(forces, _ju, -pair_forces)
            
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists * wall_dists)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
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

def _lloyd_step(c, step_size=0.5):
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    c_new = c.copy()
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if len(region) == 0 or -1 in region:
            continue
        verts = vor.vertices[region]
        if len(verts) == 0:
            continue
        centroid = np.mean(verts, axis=0)
        centroid = np.clip(centroid, -0.499999, 0.499999)
        c_new[i] = c[i] + step_size * (centroid - c[i])
        c_new[i] = np.clip(c_new[i], -0.499999, 0.499999)
    return c_new

def _lloyd_optimize(c_init, deadline, max_iters=500, step_decay=0.995):
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    step_size = 0.3
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        c_new = _lloyd_step(c, step_size)
        
        diff = c_new[_iu] - c_new[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        if np.min(dists) < 1e-6:
            c_new += np.random.normal(0, 1e-4, size=c_new.shape)
            c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            step_size *= 0.95
        step_size *= step_decay
        step_size = max(step_size, 0.001)
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=100):
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
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
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

def _active_set_polish(c_init, deadline, max_iters=1000):
    """
    Polish by moving points involved in active constraints.
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
            
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        pair_vals = dists / 2.0
        
        wall_dists = 0.5 - np.abs(c)
        
        # Find active constraints (within epsilon of min)
        eps = 1e-5
        active_pairs = np.where(pair_vals <= r + eps)[0]
        active_walls = np.where(wall_dists <= r + eps)[0]
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
            
        # Compute gradient of r w.r.t active points
        # Approximate gradient by finite differences or analytical active set grad
        # Analytical: 
        # For each active pair, force apart.
        # For each active wall, force in.
        
        grad_c = np.zeros_like(c)
        
        if len(active_pairs) > 0:
            active_diff = diff[active_pairs]
            active_dists = dists[active_pairs]
            with np.errstate(divide='ignore', invalid='ignore'):
                active_unit = active_diff / active_dists[:, np.newaxis]
            active_unit[np.isnan(active_unit)] = 0.0
            
            # Equal weight for active pairs
            weights = np.ones(len(active_pairs)) / len(active_pairs)
            
            active_forces = weights[:, np.newaxis] * active_unit
            np.add.at(grad_c, _iu[active_pairs], -active_forces)
            np.add.at(grad_c, _ju[active_pairs], active_forces)
            
        if len(active_walls) > 0:
            active_wall_indices = np.unravel_index(active_walls, wall_dists.shape)
            i_idx = active_wall_indices[0]
            k_idx = active_wall_indices[1]
            
            signs = np.sign(c[i_idx, k_idx])
            # Force inwards
            # grad = -sign * weight
            weights = np.ones(len(active_walls)) / len(active_walls)
            grad_c[i_idx, k_idx] -= signs * weights
        
        # Normalize gradient
        norm = np.linalg.norm(grad_c)
        if norm < 1e-10:
            break
            
        grad_c = grad_c / norm
        
        # Line search
        improved = False
        for step in [0.01, 0.005, 0.001, 0.0005]:
            c_try = c + step * grad_c
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
                improved = True
                break
        
        if not improved:
            # Try random perturbation of active points
            active_idx = set()
            if len(active_pairs) > 0:
                active_idx.update(_iu[active_pairs])
                active_idx.update(_ju[active_pairs])
            if len(active_walls) > 0:
                active_idx.update(active_wall_indices[0])
            
            if not active_idx:
                break
                
            c_try = c.copy()
            for i in active_idx:
                c_try[i] += np.random.normal(0, 0.001, size=3)
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
            else:
                # Stagnant, break to save time or try smaller steps
                if step < 0.0005:
                    break
                    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_bcc())
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
        
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, t_start=2.0, t_end=0.05, iters_per_temp=30)
        c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, seed_deadline, max_iters=200)
        c_polish, r_polish = _smooth_polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.005)
        c_ls, r_ls = _active_set_polish(c_polish, seed_deadline, max_iters=200)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 3.0
        
        c_lloyd2, r_lloyd2 = _lloyd_optimize(best_c, final_deadline, max_iters=100)
        if r_lloyd2 > best_r:
            best_r = r_lloyd2
            best_c = c_lloyd2.copy()
        
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.02, t_end=0.0001, iters_per_temp=50)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _active_set_polish(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c
