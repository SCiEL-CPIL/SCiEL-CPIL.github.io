"""
Optimization of 100 equal spheres in a unit cube.
Uses smooth surrogate (soft-min) formulation with L-BFGS-B to maximize minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective_and_grad(x, temperature=0.01):
    """
    Smooth surrogate: minimize -soft_min(all distances) where distances include
    pairwise/2 and wall distances. Uses log-sum-exp for numerical stability.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Compute all pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx] + 1e-15) / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all distances into one vector
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ -T * log(sum(exp(-a_i/T)))
    # Gradient: d(-soft_min)/dx = sum_j (exp(-a_j/T) / sum_k(exp(-a_k/T))) * (-da_j/dx)
    # We want to MAXIMIZE soft_min, so minimize -soft_min
    # d(-soft_min)/dx = sum_j w_j * (-da_j/dx) where w_j = exp(-a_j/T) / sum(exp(-a_k/T))
    
    scaled = -all_dists / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Objective: -soft_min = T * (log(sum_exp) + max_scaled)
    obj = temperature * (np.log(sum_exp) + max_scaled)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contribution
    # a_j = |c_i - c_j|/2 for pair (i,j)
    # da_j/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * d_ij)
    # -da_j/dc_i = -(c_i - c_j) / (2 * d_ij)
    # Contribution to grad: w_j * (-da_j/dc_i)
    
    for k in range(len(i_idx)):
        w = weights[k]
        if w < 1e-10:
            continue
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k] * 2  # actual distance
        if d < 1e-10:
            continue
        vec = centers[i] - centers[j]
        # -da/dc_i = -vec/(2*d) * 2 = -vec/d ... wait
        # a = d/2, da/dc_i = (c_i-c_j)/d * 1/2... no
        # d = |c_i - c_j|, a = d/2
        # da/dc_i = (c_i - c_j) / d * (1/2)
        # -da/dc_i = -(c_i - c_j) / (2*d)
        factor = -w / (2 * d)
        grad[i] += factor * vec
        grad[j] -= factor * vec
    
    # Wall gradient contribution
    # a = 0.5 - max(|c_x|, |c_y|, |c_z|)
    # da/dc_k = -sign(c_k) if |c_k| = max_abs
    # -da/dc_k = sign(c_k) if |c_k| = max_abs
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                grad[i, k] += w * np.sign(centers[i, k])
                break
    
    return obj, grad.flatten()

def _polish_objective_and_grad(x, active_pairs, active_walls):
    """
    High-precision polish: minimize -min over active constraints only.
    active_pairs: list of (i, j) tuples
    active_walls: list of (i, k) tuples where k is coordinate
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Collect active distances
    dists = []
    grads_info = []  # (index_in_centers, coord_or_pair, weight_info)
    
    for (i, j) in active_pairs:
        d = np.linalg.norm(centers[i] - centers[j]) / 2.0
        dists.append(d)
        grads_info.append(('pair', i, j, d))
    
    for (i, k) in active_walls:
        d = 0.5 - abs(centers[i, k])
        dists.append(d)
        grads_info.append(('wall', i, k, d))
    
    if not dists:
        return 0.0, np.zeros_like(x)
    
    min_d = np.min(dists)
    
    # Gradient: only the minimum constraint(s) contribute
    grad = np.zeros_like(centers)
    for info in grads_info:
        if info[0] == 'pair':
            _, i, j, d = info
            if np.isclose(d, min_d, atol=1e-12):
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-10:
                    factor = -1.0 / (2 * dist)
                    grad[i] += factor * vec
                    grad[j] -= factor * vec
        else:  # wall
            _, i, k, d = info
            if np.isclose(d, min_d, atol=1e-12):
                grad[i, k] += np.sign(centers[i, k])
    
    return -min_d, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # 5x4x5 = 100
    grid_centers = make_grid(5, 4, 5)
    
    # Strategy 2: Random Initialization
    rng = np.random.RandomState(42)
    random_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    # Create a set of initial guesses
    inits = [grid_centers, random_centers]
    
    # Add a few more diverse random inits
    for i in range(8):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        
        current_centers = init.copy()
        
        # Phase 1: Coarse optimization with decreasing temperature
        temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for temp in temperatures:
            if time.time() > deadline - 15.0:
                break
            
            def obj_grad(x):
                return _soft_min_objective_and_grad(x, temperature=temp)
            
            try:
                res = minimize(
                    obj_grad,
                    current_centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                current_centers = res.x.reshape(-1, 3)
            except Exception:
                pass
            
            r_curr = _compute_radius(current_centers)
            if r_curr > best_radius:
                best_radius = r_curr
                best_centers = current_centers.copy()
        
        # Phase 2: High-precision polish with active set
        if time.time() < deadline - 10.0:
            # Identify active constraints
            centers = current_centers
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            max_abs = np.max(np.abs(centers), axis=1)
            wall_dists = 0.5 - max_abs
            min_wall = np.min(wall_dists)
            
            r_curr = min(min_pair, min_wall)
            
            # Active pairs: those within 1.05 * r_curr
            i_idx, j_idx = np.triu_indices(N, k=1)
            pair_dists = dists[i_idx, j_idx] / 2.0
            active_pairs = [(i_idx[k], j_idx[k]) for k in range(len(i_idx)) if pair_dists[k] < r_curr * 1.02 + 1e-6]
            
            # Active walls
            active_walls = []
            for i in range(N):
                if wall_dists[i] < r_curr * 1.02 + 1e-6:
                    max_val = max_abs[i]
                    for k in range(3):
                        if np.isclose(abs(centers[i, k]), max_val, atol=1e-10):
                            active_walls.append((i, k))
                            break
            
            def polish_obj_grad(x):
                return _polish_objective_and_grad(x, active_pairs, active_walls)
            
            try:
                res = minimize(
                    polish_obj_grad,
                    current_centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                current_centers = res.x.reshape(-1, 3)
            except Exception:
                pass
            
            r_curr = _compute_radius(current_centers)
            if r_curr > best_radius:
                best_radius = r_curr
                best_centers = current_centers.copy()
        
        # Phase 3: Local refinement - try small perturbations
        if time.time() < deadline - 5.0 and best_centers is not None:
            for attempt in range(3):
                if time.time() > deadline - 5.0:
                    break
                # Small random perturbation
                perturbed = best_centers + rng.normal(0, 0.001, size=(N, 3))
                perturbed = np.clip(perturbed, -0.4999, 0.4999)
                
                # Quick optimize
                def obj_grad(x):
                    return _soft_min_objective_and_grad(x, temperature=0.001)
                
                try:
                    res = minimize(
                        obj_grad,
                        perturbed.flatten(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    perturbed = res.x.reshape(-1, 3)
                except Exception:
                    continue
                
                r_pert = _compute_radius(perturbed)
                if r_pert > best_radius:
                    best_radius = r_pert
                    best_centers = perturbed.copy()
    
    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
        
    return best_centers
# EVOLVE-BLOCK-END
