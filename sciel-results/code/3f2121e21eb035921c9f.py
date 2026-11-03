"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _bcc_init(n=50):
    """
    Generate initial positions on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    We generate a grid of points in a larger box, then scale to fit.
    BCC has 2 points per unit cell.
    """
    # Target: 50 points.
    # Let's use a grid of size L x L x L cells.
    # Points per cell = 2 (corners + center). But corners are shared.
    # Simpler: generate a dense grid of BCC points in a normalized space,
    # then select the best 50 and scale.
    
    # BCC lattice vectors:
    # a/2 * (1,1,1) and a/2 * (1,-1,-1) etc?
    # Standard BCC: (0,0,0) and (0.5, 0.5, 0.5) in unit cube [0,1]^3.
    
    # Let's create a 4x4x4 grid of unit cells.
    # This gives (4+1)^3 = 125 corner points and 4^3 = 64 center points?
    # No, BCC in a box [0, L]^3:
    # Points at (i, j, k) for i,j,k in 0..L-1 (corners of cells? No, lattice points)
    # And (i+0.5, j+0.5, k+0.5).
    
    # Let's just generate a large set of BCC points in a cube [0, 10]^3
    # and pick a subset, then scale to [-0.5, 0.5]^3.
    
    L = 6  # grid size
    points = []
    # Corners of the BCC unit cells (simple cubic part)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    
    # We have L^3 * 2 points. For L=6, 216 points.
    # We need 50.
    # Randomly shuffle and take 50? Or take a central chunk?
    # Taking a central chunk might be better for isotropy, but random is okay for init.
    # Let's take a random subset to avoid grid artifacts in the specific subset.
    idx = np.random.choice(len(points), N, replace=False)
    pts = points[idx]
    
    # Center and scale
    pts = pts - np.mean(pts, axis=0)
    # Scale to fit in [-0.5, 0.5]
    # Max extent is roughly L/2 * 1.732 (diagonal of cell) ?
    # Let's just normalize by max absolute value.
    max_val = np.max(np.abs(pts))
    pts = pts / max_val * 0.45  # 0.45 to leave margin from walls
    
    return pts


def _objective_and_grad(pos):
    """
    Objective: -min(r_inter, r_wall)
    We want to maximize the minimum radius.
    r = min( min_{i<j} |d_ij|/2, min_i min_k (0.5 - |c_ik|) )
    
    Using smooth approximation or just gradient of the min?
    Gradient of min is tricky. 
    Alternative: Maximize sum of logs? Or just use a repulsive potential?
    
    Let's use a potential energy approach:
    E = sum_{i<j} 1/d_ij^2 + sum_i wall_penalty
    Minimizing this tends to maximize the minimum distance.
    However, it doesn't strictly maximize the *minimum*.
    
    Better: Use a subgradient method on the objective f(pos) = -min(r).
    r = min_k r_k where r_k are constraints (distances and walls).
    grad f = - sum w_k grad r_k where w_k are weights for active constraints.
    If we only look at the *minimum* constraint, we get a subgradient.
    To make it more robust, we can use the k smallest distances.
    """
    pos = pos.reshape(N, 3)
    
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    # dist to walls: 0.5 - |x|, 0.5 - |y|, 0.5 - |z|
    wall_dists = []
    grads_wall = []
    for i in range(N):
        for axis in range(3):
            d = 0.5 - np.abs(pos[i, axis])
            wall_dists.append(d)
            # Gradient of d wrt pos[i, axis]
            # d = 0.5 - |x| -> grad = -sign(x)
            g = -np.sign(pos[i, axis])
            grad_vec = np.zeros(3)
            grad_vec[axis] = g
            grads_wall.append(grad_vec)
    
    wall_dists = np.array(wall_dists)
    grads_wall = np.array(grads_wall)
    
    # Combine all constraints
    # We want to maximize min(dist/2, wall_dist)
    # Let's define r_candidates
    # Pairwise: dist/2. Number: N*(N-1)/2
    # Wall: wall_dist. Number: 3*N
    
    # To compute gradient of the min efficiently, we identify the active set (smallest values).
    # Let's collect all values and gradients.
    # This is expensive if done naively every step.
    
    # Vectorized approach for pairwise:
    # dist/2
    r_pair = dist / 2.0
    # We need gradients for the smallest few.
    
    # Let's just compute the full gradient for the subgradient method using the single minimum.
    # Or a weighted sum of the K smallest.
    K = 5
    
    # Flatten pairwise distances
    iu = np.triu_indices(N, k=1)
    r_pair_flat = r_pair[iu]
    
    # Combine
    all_r = np.concatenate([r_pair_flat, wall_dists])
    
    # Find indices of K smallest
    # argsort is O(M log M). M ~ 1225 + 150 = 1375. Fast.
    sorted_idx = np.argsort(all_r)
    active_idx = sorted_idx[:K]
    
    # Compute gradient
    grad = np.zeros((N, 3))
    
    # Weights: uniform for active? Or inverse distance?
    # Simple: uniform weight for active constraints.
    w = 1.0 / K
    
    for idx in active_idx:
        if idx < len(r_pair_flat):
            # Pairwise constraint
            i, j = iu[0][idx], iu[1][idx]
            # r = |p_i - p_j| / 2
            # grad r wrt p_i = (p_i - p_j) / |p_i - p_j|
            # grad r wrt p_j = (p_j - p_i) / |p_j - p_i|
            d_vec = pos[i] - pos[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-8:
                continue
            unit = d_vec / d_norm
            grad[i] += w * unit
            grad[j] -= w * unit
        else:
            # Wall constraint
            idx_wall = idx - len(r_pair_flat)
            i = idx_wall // 3
            axis = idx_wall % 3
            grad[i] += w * grads_wall[idx_wall]
    
    # We want to maximize min(r), so we minimize -min(r).
    # Gradient of -min(r) is - (subgradient of min(r))
    # So we return -grad as the gradient for minimization.
    return -np.min(all_r), -grad.flatten()


def _optimize(pos0, time_limit=100.0):
    """
    Optimize the positions using gradient ascent (via minimization of negative radius).
    """
    start_time = time.time()
    pos = pos0.copy()
    pos_flat = pos.flatten()
    
    # Simple gradient ascent with adaptive step size
    step = 0.05
    best_r = -1e9
    best_pos = pos.copy()
    
    # Compute initial r
    r, grad = _objective_and_grad(pos_flat)
    best_r = -r
    best_pos = pos.copy()
    
    while time.time() - start_time < time_limit:
        # Gradient ascent step
        # pos_new = pos + step * grad
        # Note: grad from _objective_and_grad is for minimizing -r, so it points in direction of decreasing -r (increasing r).
        # Wait. 
        # f(pos) = -min(r). 
        # We want to minimize f.
        # grad f = - subgrad(min r).
        # Descent step: pos_new = pos - step * grad f = pos + step * subgrad(min r).
        # My function returns -grad(min r) as the gradient.
        # So if I do pos - step * returned_grad, I am doing pos + step * subgrad(min r). Correct.
        
        pos_flat = pos_flat - step * grad
        
        # Clip to cube? No, the objective handles walls. But if it goes outside, r becomes negative.
        # It's better to allow it to go outside and let the penalty pull it back, 
        # but if it goes too far, gradients might vanish or become unstable.
        # Let's project to [-0.5, 0.5] if it goes outside?
        # Actually, the wall distance is 0.5 - |x|. If x > 0.5, distance is negative.
        # The gradient is -sign(x). If x > 0.5, grad is -1. 
        # Step: pos - step*(-1) = pos + step. It moves further out!
        # This is a problem. The gradient of the distance function 0.5-|x| points INWARD for x>0.5?
        # d/dx (0.5 - |x|) = -sign(x). For x>0, grad is -1.
        # We want to increase the distance. So we should move in direction of grad.
        # My update rule: pos_new = pos + step * subgrad(r).
        # If r is wall distance, subgrad is -sign(x).
        # If x > 0.5, sign(x)=1, subgrad=-1.
        # pos_new = x + step*(-1) = x - step. It moves INWARD. Correct.
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(pos_flat)):
            pos_flat = best_pos.flatten()
            break
            
        # Recompute objective and gradient
        r, grad = _objective_and_grad(pos_flat)
        current_r = -r
        
        if current_r > best_r:
            best_r = current_r
            best_pos = pos_flat.reshape(N, 3).copy()
            # Increase step if improving
            step *= 1.05
            if step > 0.1: step = 0.1
        else:
            # Decrease step if not improving
            step *= 0.95
            if step < 1e-5: step = 1e-5
            
        # Optional: Random restarts if stuck
        # For now, just keep going.
        
    return best_pos, best_r


def construct():
    """
    Construct the optimal packing.
    """
    start_time = time.time()
    total_time = 140.0  # Safety margin for 150s limit
    
    # 1. Initialize with BCC
    pos0 = _bcc_init()
    
    # 2. Optimize
    pos, r = _optimize(pos0, time_limit=total_time - (time.time() - start_time))
    
    # Check validity
    if not np.all(np.isfinite(pos)):
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return pos
# EVOLVE-BLOCK-END
