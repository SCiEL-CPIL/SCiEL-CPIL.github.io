"""Combined solver: multi-start + repulsion annealing + softmin gradient + local polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 50

# Precompute upper triangle indices
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """Compute the largest common radius for centers c (N,3) in [-0.5,0.5]^3."""
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _softmin_value_and_grad(c, t):
    """
    Compute softmin of all constraints (pair distances/2 and wall distances)
    and its gradient. Returns (softmin_value, gradient).
    We maximize softmin, so we minimize -softmin.
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    
    # All values: pairs and all wall distances
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    
    # Softmin: -t * log(sum(exp(-x/t)))
    # Numerically stable
    max_neg = np.max(-all_vals / t)
    exp_terms = np.exp(-all_vals / t - max_neg)
    total = np.sum(exp_terms)
    
    if total < 1e-300:
        return np.min(all_vals), np.zeros_like(c)
    
    softmin_val = -t * (np.log(total) + max_neg)
    
    # Gradient of -softmin = gradient of t * log(sum(exp(-x/t)))
    # d/dx_i [t * log(S)] where S = sum_j exp(-x_j/t)
    # = t * (1/S) * sum_j exp(-x_j/t) * (-1/t) * dx_j/dx_i
    # = -(1/S) * sum_j exp(-x_j/t) * dx_j/dx_i
    
    # Weights
    w_pair = exp_terms[:_NUM_PAIRS] / total  # weights for pair distances
    w_wall = exp_terms[_NUM_PAIRS:] / total  # weights for wall distances
    
    grad = np.zeros_like(c)
    
    # Pair contributions: d_ij = |c_i - c_j|
    # d(pair_val)/d(c_i) = 0.5 * (c_i - c_j) / d_ij
    # d(pair_val)/d(c_j) = 0.5 * (c_j - c_i) / d_ij
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # grad contribution from pairs: -sum_j w_j * dx_j/dx_i
    # For pair (i,j): dx/d(c_i) = 0.5 * unit_vec, dx/d(c_j) = -0.5 * unit_vec
    # So: grad[c_i] -= w_pair * 0.5 * unit_vec
    #     grad[c_j] += w_pair * 0.5 * unit_vec
    pair_grad = 0.5 * w_pair[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _iu[idx], _ju[idx]
        grad[i] -= pair_grad[idx]
        grad[j] += pair_grad[idx]
    
    # Wall contributions: wall_val = 0.5 - |c_ik|
    # d/d(c_ik) = -sign(c_ik)
    # grad[c_ik] -= w_wall * (-sign(c_ik)) = w_wall * sign(c_ik)
    wall_w = w_wall.reshape(N, 3)
    signs = np.sign(c)
    grad += wall_w * signs
    
    return softmin_val, grad

def _seed_fcc():
    """FCC-inspired initial configuration."""
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
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
    """BCC-inspired initial configuration."""
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

def _seed_grid():
    """Grid-based initial configuration."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_anneal(c_init, deadline, t_start=1.0, t_end=0.05, total_iters=300):
    """Repulsion-based simulated annealing to spread out spheres."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.geomspace(t_start, t_end, total_iters)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        # Compute repulsive forces
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            force_mag = 1.0 / (dists * dists)
        force_mag = np.clip(force_mag, 0, 1e4)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[:, np.newaxis]
        unit_vecs[np.isnan(unit_vecs)] = 0.0
        
        pair_forces = force_mag[:, np.newaxis] * unit_vecs
        forces = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _iu[idx], _ju[idx]
            forces[i] += pair_forces[idx]
            forces[j] -= pair_forces[idx]
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        with np.errstate(divide='ignore', invalid='ignore'):
            wall_force = 1.0 / (wall_dists * wall_dists)
        wall_force = np.clip(wall_force, 0, 1e4)
        wall_force[np.isnan(wall_force)] = 0
        signs = np.sign(c)
        forces += wall_force * signs
        
        # Update with noise
        noise = np.random.normal(0, t * 0.05, size=c.shape)
        c_new = c + forces * 0.005 + noise
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        
        # Metropolis acceptance
        r_cur = _compute_radius(c)
        delta = r_new - r_cur
        if delta > 0 or np.random.rand() < np.exp(delta / t):
            c = c_new
    
    return best_c, best_r

def _gradient_optimize(c_init, deadline, t0=5.0, t_final=0.5, max_iter=200):
    """Gradient-based optimization with annealing temperature."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        t = t0 * (t_final / t0) ** (it / max_iter)
        val, grad = _softmin_value_and_grad(c, t)
        
        # Gradient ascent on softmin (we want to maximize)
        lr = 0.01
        c_new = c + lr * grad
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        
        # Accept with Metropolis
        r_cur = _compute_radius(c)
        delta = r_new - r_cur
        if delta > 0 or np.random.rand() < np.exp(delta / (t * 0.1)):
            c = c_new
    
    return best_c, best_r

def _local_polish(c_init, deadline):
    """Local refinement using Nelder-Mead on the negative radius."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def objective(x):
        c = x.reshape(-1, 3)
        return -_compute_radius(c)
    
    try:
        result = minimize(objective, c.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
        candidate = result.x.reshape(-1, 3)
        candidate = np.clip(candidate, -0.4999, 0.4999)
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_c = candidate
    except Exception:
        pass
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    np.random.seed(42)
    
    # Generate diverse initial configurations
    seeds = [
        _seed_fcc(),
        _seed_bcc(),
        _seed_grid(),
        _seed_random(),
    ]
    
    best_centres = None
    best_radius = -np.inf
    
    for seed_idx, c_init in enumerate(seeds):
        if time.time() > deadline - 30.0:
            break
        
        # Phase 1: Repulsion annealing
        c1, r1 = _repulsion_anneal(c_init, deadline - 20.0, 
                                     t_start=1.0, t_end=0.1, total_iters=150)
        
        if r1 > best_radius:
            best_radius = r1
            best_centres = c1.copy()
        
        # Phase 2: Gradient optimization with softmin
        if time.time() < deadline - 15.0:
            c2, r2 = _gradient_optimize(c1, deadline - 10.0,
                                         t0=5.0, t_final=0.5, max_iter=100)
            if r2 > best_radius:
                best_radius = r2
                best_centres = c2.copy()
        
        # Phase 3: Local polish
        if time.time() < deadline - 5.0:
            c3, r3 = _local_polish(best_centres, deadline - 2.0)
            if r3 > best_radius:
                best_radius = r3
                best_centres = c3.copy()
    
    # Final: if we still have time, do one more polish on the best
    if time.time() < deadline - 2.0 and best_centres is not None:
        c_final, r_final = _local_polish(best_centres, deadline - 1.0)
        if r_final > best_radius:
            best_radius = r_final
            best_centres = c_final
    
    if best_centres is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
