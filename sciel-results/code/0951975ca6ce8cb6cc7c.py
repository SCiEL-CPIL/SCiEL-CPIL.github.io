"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Uses a combination of repulsion annealing, Lloyd's algorithm, 
and gradient-based optimization on a soft-min objective.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

def _compute_radius(centers):
    """Calculate the exact packing radius."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin_radius_and_grad(centers, beta=50.0):
    """
    Compute soft-min approximation of the packing radius and its gradient.
    r_soft = -1/beta * log(sum(exp(-beta * d_i)))
    where d_i are the constraint distances (pairwise/2 and wall).
    """
    centers = np.asarray(centers).reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # We want min dist / 2. Let's work with squared distances for stability in exp?
    # Actually, exp(-beta * dist) is better for gradient.
    # Let's compute actual distances.
    dists = np.sqrt(dists_sq + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Combine all constraints into a flat array
    # We need gradients w.r.t centers.
    # This is complex to vectorize fully for gradient, so we'll compute objective and grad separately or use numerical?
    # Analytical grad is better.
    
    # Let's define the constraints explicitly for gradient computation.
    # Constraint 1: Pairwise d_ij / 2
    # Constraint 2: Wall w_i
    
    # Objective: F = -1/beta * log( sum_{pairs} exp(-beta * d_ij/2) + sum_{walls} exp(-beta * w_i) )
    # Grad F = -1/beta * 1/S * sum_k (-beta * grad(d_k)) exp(-beta * d_k)
    #        = 1/S * sum_k exp(-beta * d_k) * grad(d_k)
    # where S = sum_k exp(-beta * d_k)
    
    # To avoid underflow/overflow, let's use the log-sum-exp trick if beta is large.
    # But beta=50 is moderate.
    
    # Let's compute the terms.
    # Pairwise terms
    # d_ij = ||c_i - c_j|| / 2
    # grad_{c_i} d_ij = (c_i - c_j) / (2 * ||c_i - c_j||)
    
    # Wall terms
    # w_i = min_k (0.5 - |c_ik|)
    # This is also a min, so we need soft-min for walls too? Or just take the min wall distance?
    # Usually, it's safer to soft-min over all walls too, or just use the min wall distance in the set.
    # Let's use the min wall distance for each sphere as a constraint.
    
    # Let's simplify: Use soft-min over pairwise distances AND min-wall-distance per sphere.
    
    # 1. Pairwise soft-min
    # d_pairs = dists / 2.0
    # exp_terms_pairs = exp(-beta * d_pairs)
    # Sum_pairs = sum(exp_terms_pairs)
    
    # 2. Wall soft-min (per sphere, then global? or just global min?)
    # If we just use global min wall, it's non-smooth.
    # Let's use soft-min over all wall distances as well.
    # w_all = 0.5 - np.abs(centers).flatten()
    # exp_terms_walls = exp(-beta * w_all)
    # Sum_walls = sum(exp_terms_walls)
    
    # Total Sum S = Sum_pairs + Sum_walls
    # Obj = -1/beta * log(S)
    
    # Gradient:
    # grad = 1/S * ( Sum_pairs_grad + Sum_walls_grad )
    
    # Sum_pairs_grad:
    # For each pair (i,j), term is exp(-beta * d_ij/2).
    # grad_{c_i} term = exp(...) * (-beta/2) * grad_{c_i} d_ij
    # grad_{c_i} d_ij = (c_i - c_j) / (2 * ||c_i - c_j||)  -- Wait, d_ij = ||c_i - c_j|| / 2.
    # grad_{c_i} ||c_i - c_j|| = (c_i - c_j) / ||c_i - c_j||
    # So grad_{c_i} d_ij = (c_i - c_j) / (2 * ||c_i - c_j||)
    # Thus grad_{c_i} term = exp(...) * (-beta/2) * (c_i - c_j) / (2 * ||c_i - c_j||)
    #                     = -beta/4 * exp(...) * (c_i - c_j) / ||c_i - c_j||
    
    # Let's compute this.
    
    # Initialize gradient
    grad = np.zeros((N, 3))
    
    # Pairwise part
    # dists is N x N
    # We need exp(-beta * dists / 2)
    # To avoid overflow, we can subtract max? No, beta is positive, exp is small.
    # But if dists are small, exp is close to 1.
    
    # Let's compute the weights w_ij = exp(-beta * d_ij / 2)
    # d_ij = dists / 2
    # w_ij = exp(-beta * dists / 4)
    
    # We need to sum over j for each i.
    # grad_{c_i} S_pairs = sum_{j != i} grad_{c_i} (exp(-beta * d_ij / 2))
    # = sum_{j != i} exp(-beta * d_ij / 2) * (-beta/2) * (c_i - c_j) / (2 * ||c_i - c_j||)
    # = -beta/4 * sum_{j != i} w_ij * (c_i - c_j) / ||c_i - c_j||
    
    # Let's vectorize.
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # w = np.exp(-beta * dists / 4.0)
    # np.fill_diagonal(w, 0) # Exclude self
    
    # term = w * (diff / (dists + 1e-12))[:, :, np.newaxis]
    # sum_term = np.sum(term, axis=1)
    # grad_pairs = -beta / 4.0 * sum_term
    
    # Wall part
    # S_walls = sum_{i,k} exp(-beta * (0.5 - |c_ik|))
    # grad_{c_i} S_walls = sum_k exp(-beta * w_ik) * (-beta) * grad_{c_i} w_ik
    # w_ik = 0.5 - |c_ik|
    # grad_{c_i} w_ik = -sign(c_ik)
    # So grad_{c_i} S_walls = sum_k exp(-beta * w_ik) * (-beta) * (-sign(c_ik))
    # = beta * sum_k w_ik * sign(c_ik)
    
    # Let's compute.
    # w_vals = 0.5 - np.abs(centers)
    # exp_w = np.exp(-beta * w_vals)
    # signs = np.sign(centers)
    # grad_walls = beta * np.sum(exp_w * signs, axis=1, keepdims=True)
    
    # Total S = S_pairs + S_walls
    # Obj = -1/beta * log(S)
    # Grad Obj = 1/S * (Grad S_pairs + Grad S_walls)
    
    # Let's implement this carefully.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Weights for pairs
    # d_ij = dists / 2
    # w_ij = exp(-beta * d_ij) = exp(-beta * dists / 2)
    # Wait, earlier I said d_ij = dists/2, so exp(-beta * d_ij) = exp(-beta * dists / 2).
    # My previous derivation had /4. Let's re-verify.
    # d_ij = ||c_i - c_j|| / 2.
    # exp(-beta * d_ij) = exp(-beta * ||c_i - c_j|| / 2).
    # grad_{c_i} d_ij = (c_i - c_j) / (2 * ||c_i - c_j||).
    # grad_{c_i} exp(-beta * d_ij) = exp(...) * (-beta) * (c_i - c_j) / (2 * ||c_i - c_j||).
    # So coefficient is -beta/2.
    
    w_pairs = np.exp(-beta * dists / 2.0)
    np.fill_diagonal(w_pairs, 0)
    
    # grad_pairs = -beta / 2.0 * sum_j w_ij * (c_i - c_j) / ||c_i - c_j||
    # diff / dists is the unit vector.
    # Note: dists is inf on diagonal, but w is 0, so 0/inf is 0? No, 0*inf is nan.
    # So we must mask.
    
    unit_vecs = diff / (dists + 1e-12)[:, :, np.newaxis]
    # Mask diagonal
    mask = np.ones_like(dists, dtype=bool)
    np.fill_diagonal(mask, False)
    unit_vecs[~mask] = 0
    
    grad_pairs = -beta / 2.0 * np.sum(w_pairs[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Walls
    w_vals = 0.5 - np.abs(centers)
    w_walls = np.exp(-beta * w_vals)
    signs = np.sign(centers)
    # If c_ik is 0, sign is 0. This is a point of non-differentiability.
    # For small beta, it's fine.
    grad_walls = beta * np.sum(w_walls * signs, axis=1, keepdims=True)
    
    S_pairs = np.sum(w_pairs)
    S_walls = np.sum(w_walls)
    S = S_pairs + S_walls
    
    if S < 1e-300:
        return 0.0, grad
    
    obj = -1.0 / beta * np.log(S)
    grad = (grad_pairs + grad_walls) / S
    
    return obj, grad

def _repulsion_anneal(centers, time_limit=10.0):
    """
    Anneal a repulsive potential to spread points.
    """
    start_time = time.time()
    centers = centers.copy()
    beta = 10.0
    step = 0.01
    
    while time.time() - start_time < time_limit:
        # Compute forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0
        
        # Force = -grad U. U = sum 1/d. Force = 1/d^2 * unit_vec.
        # We want to move points apart.
        # grad_{c_i} (1/d_ij) = -1/d_ij^2 * (c_i - c_j)/d_ij = -(c_i - c_j)/d_ij^3
        # So force on i is sum_j (c_i - c_j)/d_ij^3.
        
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = dists[i, j]
                if d < 1e-6:
                    # Random kick
                    forces[i] += np.random.normal(0, 1, 3) * 0.1
                else:
                    forces[i] += (centers[i] - centers[j]) / (d**3)
        
        # Wall repulsion
        # Push away from walls if close
        wall_dist = 0.5 - np.abs(centers)
        # If wall_dist < 0.1, push in
        for i in range(N):
            for k in range(3):
                if wall_dist[i, k] < 0.1:
                    # Push towards center
                    forces[i, k] += -np.sign(centers[i, k]) * (0.1 - wall_dist[i, k]) * 10.0
        
        # Update
        centers += forces * step
        centers = np.clip(centers, -0.49, 0.49)
        
        # Cool down
        step *= 0.95
        if step < 0.001:
            break
            
    return centers

def _lloyd_step(centers):
    """
    One step of Lloyd's algorithm: move each point to the centroid of its Voronoi cell.
    """
    try:
        vor = Voronoi(centers)
    except Exception:
        return centers
    
    new_centers = centers.copy()
    for i in range(N):
        # Get region
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            continue
            
        # Get vertices
        verts = vor.vertices[region]
        
        # Clip vertices to cube
        verts = np.clip(verts, -0.5, 0.5)
        
        # Calculate centroid
        if len(verts) > 0:
            centroid = np.mean(verts, axis=0)
            new_centers[i] = centroid
    
    # Ensure no duplicates
    # If two points are very close, merge? No, just let it be.
    
    return new_centers

def _lloyd_optimize(centers, time_limit=10.0):
    """
    Run Lloyd's algorithm for a set number of iterations or time limit.
    """
    start_time = time.time()
    centers = centers.copy()
    
    while time.time() - start_time < time_limit:
        new_centers = _lloyd_step(centers)
        # Check if converged
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
        time.sleep(0.01) # Small sleep to avoid CPU spike if Voronoi is fast
        
    return centers

def _gradient_optimize(centers, time_limit=20.0):
    """
    Use L-BFGS-B to minimize the negative soft-min objective.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        c = x.reshape(N, 3)
        obj, grad = _softmin_radius_and_grad(c, beta=50.0)
        return -obj, -grad.reshape(N * 3)
    
    x0 = centers.reshape(N * 3)
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    res = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if res.success or res.fun < -_softmin_radius_and_grad(centers, 50.0)[0]:
        return res.x.reshape(N, 3)
    else:
        return centers

def _local_search(centers, time_limit=10.0):
    """
    Final local search on exact radius.
    """
    start_time = time.time()
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    iu = np.triu_indices(N, k=1)
    step_size = 0.01
    
    while time.time() - start_time < time_limit:
        # Find closest pair
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        # Find closest wall
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k = min_wall_idx[0]
        
        # Try moving i and j apart
        # Direction from j to i
        if i != j:
            dir_ij = (centers[i] - centers[j]) / (np.linalg.norm(centers[i] - centers[j]) + 1e-10)
            # Move i along dir_ij, j along -dir_ij
            new_i = centers[i] + dir_ij * step_size
            new_j = centers[j] - dir_ij * step_size
            
            # Clip
            new_i = np.clip(new_i, -0.495, 0.495)
            new_j = np.clip(new_j, -0.495, 0.495)
            
            old_i, old_j = centers[i], centers[j]
            centers[i], centers[j] = new_i, new_j
            
            r = _compute_radius(centers)
            if r > current_r:
                current_r = r
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                centers[i], centers[j] = old_i, old_j
                step_size *= 0.9
        
        # Try moving k away from wall
        # Direction from wall to center
        # Find which axis is closest
        axis = min_wall_idx[1]
        # If c_ka < 0, push right (+), else push left (-)
        dir_k = -np.sign(centers[k, axis])
        dir_k_vec = np.zeros(3)
        dir_k_vec[axis] = dir_k
        
        new_k = centers[k] + dir_k_vec * step_size
        new_k = np.clip(new_k, -0.495, 0.495)
        
        old_k = centers[k]
        centers[k] = new_k
        
        r = _compute_radius(centers)
        if r > current_r:
            current_r = r
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        else:
            centers[k] = old_k
            step_size *= 0.95
        
        # Random perturbation to escape local minima
        if np.random.rand() < 0.05:
            idx = np.random.randint(N)
            perturbation = np.random.normal(0, step_size, 3)
            new_pos = centers[idx] + perturbation
            new_pos = np.clip(new_pos, -0.495, 0.495)
            old_pos = centers[idx]
            centers[idx] = new_pos
            r = _compute_radius(centers)
            if r > current_r:
                current_r = r
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                centers[idx] = old_pos
                
        step_size = max(step_size, 0.001)
        
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initialization
    # Use a grid-like initialization to start with good separation
    # 50 points. 3x4x5 = 60. Take first 50.
    m = [3, 4, 5]
    ticks = [np.linspace(-0.4, 0.4, n) for n in m]
    grid = np.array([(x, y, z) for z in ticks[2] for y in ticks[1] for x in ticks[0]])
    # Shuffle to break symmetry
    np.random.shuffle(grid)
    centers = grid[:N]
    
    # 1. Repulsion Annealing
    time_left = deadline - time.time()
    if time_left > 5:
        centers = _repulsion_anneal(centers, time_limit=min(5.0, time_left - 1.0))
    
    # 2. Lloyd's Algorithm
    time_left = deadline - time.time()
    if time_left > 10:
        centers = _lloyd_optimize(centers, time_limit=min(10.0, time_left - 1.0))
    
    # 3. Gradient Optimization (Soft-Min)
    time_left = deadline - time.time()
    if time_left > 20:
        centers = _gradient_optimize(centers, time_limit=min(20.0, time_left - 1.0))
    
    # 4. Final Local Search
    time_left = deadline - time.time()
    if time_left > 10:
        centers = _local_search(centers, time_limit=min(10.0, time_left - 1.0))
    
    # Ensure valid
    centers = np.clip(centers, -0.495, 0.495)
    
    return centers

# EVOLVE-BLOCK-END
