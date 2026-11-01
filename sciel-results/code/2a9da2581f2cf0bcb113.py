import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

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
                # BCC: corner + center
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

def _maxmin_radius_and_grad(c, tol=1e-6):
    """
    Compute the max-min radius and its gradient.
    The objective is r = min(min_pair_dist/2, min_wall_dist).
    The gradient is the average of the gradients of the active constraints.
    """
    # 1. Compute all pair distances and wall distances
    diff = c[_iu] - c[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    wall_vals_flat = wall_vals.ravel()
    wall_indices = np.arange(N * 3).reshape(N, 3)

    # 2. Find the minimum value
    min_pair_val = np.min(pair_vals)
    min_wall_val = np.min(wall_vals_flat)
    
    r = min(min_pair_val, min_wall_val)
    
    # 3. Identify active constraints (within tolerance)
    active_pairs = np.where(np.abs(pair_vals - min_pair_val) < tol)[0]
    active_walls = np.where(np.abs(wall_vals_flat - min_wall_val) < tol)[0]
    
    # We need to handle the case where both pair and wall mins are close to r.
    # Actually, r is the global min. So active constraints are those equal to r.
    # Let's refine:
    active_pairs = np.where(np.abs(pair_vals - r) < tol)[0]
    active_walls = np.where(np.abs(wall_vals_flat - r) < tol)[0]
    
    grad_c = np.zeros_like(c)
    count = 0
    
    # 4. Accumulate gradients for active pair constraints
    if len(active_pairs) > 0:
        # For each active pair (i, j), grad of dist/2 w.r.t c_i is (c_i - c_j) / dist
        # grad w.r.t c_j is (c_j - c_i) / dist = -(c_i - c_j) / dist
        for idx in active_pairs:
            i, j = _PAIR_IDX[idx]
            d = dists[idx]
            if d < 1e-10:
                # Degenerate case, add random noise
                vec = np.random.normal(size=3)
                vec /= np.linalg.norm(vec)
            else:
                vec = diff[idx] / d
            
            grad_c[i] += 0.5 * vec  # derivative of dist/2 is 0.5 * unit_vec
            grad_c[j] -= 0.5 * vec
            count += 1
            
    # 5. Accumulate gradients for active wall constraints
    if len(active_walls) > 0:
        for w_idx in active_walls:
            i = w_idx // 3
            k = w_idx % 3
            # wall_val = 0.5 - |c_ik|
            # grad w.r.t c_ik is -sign(c_ik)
            # To maximize r, we move in direction of gradient.
            # If c_ik > 0, wall_val = 0.5 - c_ik, grad = -1. Move in -x direction.
            # If c_ik < 0, wall_val = 0.5 + c_ik, grad = +1. Move in +x direction.
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0 # arbitrary
            
            grad_c[i, k] -= sign
            count += 1
            
    # 6. Normalize gradient
    if count > 0:
        grad_c /= count
    else:
        # If no active constraints (should not happen), return zero
        grad_c = np.zeros_like(c)
        
    return r, grad_c

def _repulsion_anneal(c_init, deadline, t_start=0.5, t_end=0.01, iters_per_temp=20):
    """
    Repulsion-based simulated annealing.
    Uses vectorized force calculation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 10)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces (vectorized)
            forces = np.zeros_like(c)
            
            # Pair repulsion: F ~ 1/d^2
            diff = c[_iu] - c[_ju]
            dists_sq = np.sum(diff * diff, axis=-1)
            dists = np.sqrt(dists_sq)
            
            # Avoid division by zero
            dists_safe = np.where(dists < 1e-6, 1e-6, dists)
            force_mag = 1.0 / (dists_safe * dists_safe)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            unit_vecs = diff / dists_safe[:, np.newaxis]
            
            # Accumulate forces
            # forces[i] += F * u, forces[j] -= F * u
            np.add.at(forces, _iu, force_mag[:, np.newaxis] * unit_vecs)
            np.add.at(forces, _ju, -force_mag[:, np.newaxis] * unit_vecs)
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            wall_dists_safe = np.where(wall_dists < 1e-6, 1e-6, wall_dists)
            wall_force_mag = 1.0 / (wall_dists_safe * wall_dists_safe)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            
            signs = np.sign(c)
            # Force pushes away from wall
            # If c > 0, wall is at 0.5, dist = 0.5-c. Force should be negative (towards -x)?
            # No, repulsion from wall means if close to 0.5, push towards -x.
            # Gradient of (0.5 - c)^-2 is 2(0.5-c)^-3.
            # Let's just use sign to push inwards.
            # If c > 0, push -x. If c < 0, push +x.
            forces -= wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.05, size=c.shape)
            c_new = c + forces * 0.005 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            # Accept if better or with probability
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            r_curr = _compute_radius(c)
            delta_r = r_new - r_curr
            if delta_r > 0 or np.random.random() < np.exp(delta_r / max(t, 1e-10)):
                c = c_new
                
    return best_c, best_r

def _polish_maxmin(c_init, deadline, maxiter=200, tol=1e-12):
    """
    Polish using L-BFGS-B on the max-min objective with explicit gradient.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def obj(x):
        c_x = x.reshape(N, 3)
        r, _ = _maxmin_radius_and_grad(c_x, tol=1e-8)
        return -r
        
    def grad(x):
        c_x = x.reshape(N, 3)
        _, g = _maxmin_radius_and_grad(c_x, tol=1e-8)
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
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14}
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

def _local_search_maxmin(c_init, deadline, max_iters=200):
    """
    Local search: Perturb points that are constraining the radius.
    Uses max-min gradient for direction.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r, grad = _maxmin_radius_and_grad(c, tol=1e-6)
        
        if r <= best_r:
            c = best_c.copy()
            r, grad = _maxmin_radius_and_grad(c, tol=1e-6)
            
        # Try moving in gradient direction with varying steps
        # Gradient points in direction of increasing r
        moved = False
        for step in (0.005, 0.01, 0.02, 0.001):
            c_try = c + step * grad
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r + 1e-12:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
                moved = True
                break
        
        if not moved:
            # If no improvement, try a small random perturbation
            c_try = c + np.random.normal(0, 0.001, size=c.shape)
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
            else:
                # Stuck, break or continue with smaller noise
                if it > 50:
                    break

    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_bcc())
    seeds.append(_seed_random())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time) / n_seeds
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
        
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - 10.0)
        
        # Stage 1: Repulsion Annealing (Global)
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, t_start=0.5, t_end=0.05, iters_per_temp=10)
        
        # Stage 2: Max-Min Polish (Local)
        c_polish, r_polish = _polish_maxmin(c_anneal, seed_deadline, maxiter=100)
        
        # Stage 3: Local Search (Fine-tune)
        c_ls, r_ls = _local_search_maxmin(c_polish, seed_deadline, max_iters=100)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 5.0
        
        # More aggressive polish
        c_final, r_final = _polish_maxmin(best_c, final_deadline, maxiter=300, tol=1e-14)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Final local search
        if time.time() < final_deadline - 5.0:
            c_ls, r_ls = _local_search_maxmin(best_c, final_deadline, max_iters=200)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()

    return best_c
