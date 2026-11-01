import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(positions):
    """
    Calculate the score for a given set of positions.
    Returns the minimum radius such that all circles fit.
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    radii = np.sqrt(np.sum(positions**2, axis=1))
    min_boundary_dist = 1.0 - np.max(radii)
    
    r = min(min_dist / 2.0, min_boundary_dist)
    return r

def _soft_min_obj(x, beta):
    """
    Objective function for optimization: minimize -soft_min(distances)
    x is a flattened array of shape (2N,)
    """
    pos = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    radii = np.sqrt(np.sum(pos**2, axis=1))
    boundary = 1.0 - radii
    
    # Combine all distances into one array
    # We want to maximize the minimum of:
    # 1. d_ij / 2
    # 2. 1 - |c_i|
    
    # For soft min, we use log-sum-exp trick
    # soft_min(A) = -1/beta * log(sum(exp(-beta * A)))
    # We want to maximize soft_min, so we minimize -soft_min
    
    # Distances to consider:
    # All pairwise distances / 2
    # All boundary distances
    
    # Get unique pairs to avoid double counting (optional, but cleaner)
    # Actually, for soft min, double counting just weights those more, which is fine
    # Or we can just use all off-diagonal elements
    
    # Let's flatten the distance matrix (excluding diagonal)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    # Combine with boundary distances
    all_dists = np.concatenate([pair_dists, boundary])
    
    # Soft min
    # To avoid overflow, subtract max
    max_d = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - max_d)))) + (-beta * max_d)
    soft_min_val = -log_sum / beta
    
    # We want to maximize soft_min, so return negative
    return -soft_min_val

def _soft_min_grad(x, beta):
    """
    Gradient of the soft min objective.
    """
    pos = x.reshape(N, 2)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    radii = np.sqrt(np.sum(pos**2, axis=1))
    boundary = 1.0 - radii
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    all_dists = np.concatenate([pair_dists, boundary])
    
    # Soft min gradient:
    # d(soft_min)/d(a_k) = -1/sum(exp(-beta*a)) * exp(-beta*a_k) * (-beta)
    #                      = exp(-beta*a_k) / sum(exp(-beta*a_j))
    # Let p_k = exp(-beta*a_k) / sum(exp(-beta*a_j))
    # Then d(soft_min)/d(a_k) = p_k
    
    # We are minimizing -soft_min, so gradient is -p_k
    
    max_d = np.max(all_dists)
    exp_neg = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_neg)
    p = exp_neg / sum_exp  # Probabilities
    
    # Gradient of -soft_min with respect to all_dists is -p
    # Now we need to backpropagate to x
    
    grad_x = np.zeros_like(x)
    grad_pos = np.zeros((N, 2))
    
    # Gradient from pairwise distances
    # a_k = dist[i,j] / 2
    # d(a_k)/d(pos[i]) = (pos[i] - pos[j]) / dist[i,j] * 1/2
    # d(a_k)/d(pos[j]) = (pos[j] - pos[i]) / dist[i,j] * 1/2
    
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w = -p[k]  # Weight for this distance
        
        # Gradient contribution to pos[i] and pos[j]
        vec = pos[i] - pos[j]
        d = dist[i, j]
        if d > 1e-10:
            grad_pos[i] += w * vec / (2 * d)
            grad_pos[j] += w * (-vec) / (2 * d)
    
    # Gradient from boundary distances
    # a_k = 1 - |pos[i]|
    # d(a_k)/d(pos[i]) = -pos[i] / |pos[i]|
    
    for i in range(N):
        w = -p[len(pair_dists) + i]  # Weight for boundary distance of point i
        
        r = radii[i]
        if r > 1e-10:
            grad_pos[i] += w * (-pos[i] / r)
    
    grad_x = grad_pos.flatten()
    return grad_x

def _generate_initial_config():
    """
    Generate an initial configuration using a hexagonal lattice.
    """
    s = np.sqrt(np.pi / (N * np.sqrt(3)/2)) * 0.9
    
    points = []
    k_max = int(1.0 / s) + 2
    
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - s/2.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        # Fallback to random
        points = np.random.randn(N, 2) * 0.5
    
    idx = np.random.choice(len(points), N, replace=False)
    return points[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_score = -np.inf
    
    # Try multiple initializations
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        pos = _generate_initial_config()
        x0 = pos.flatten()
        
        # Use L-BFGS-B for optimization
        # We can use a moderate beta for soft min
        beta = 50.0
        
        # Define objective and gradient
        def obj(x):
            return _soft_min_obj(x, beta)
        
        def grad(x):
            return _soft_min_grad(x, beta)
        
        try:
            res = minimize(
                obj, 
                x0, 
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            x_opt = res.x
            pos_opt = x_opt.reshape(N, 2)
            
            # Check validity
            if np.all(np.isfinite(pos_opt)):
                score = _score(pos_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
        except Exception:
            continue
    
    # If no valid result from optimization, use best from initializations
    if best_pos is None:
        # Generate a valid configuration
        pos = _generate_initial_config()
        # Ensure it's inside the unit circle
        radii = np.sqrt(np.sum(pos**2, axis=1))
        mask = radii > 1.0
        if np.any(mask):
            pos[mask] = pos[mask] / radii[mask, np.newaxis] * 0.9
        best_pos = pos
        best_score = _score(best_pos)
    
    # Final polish with small perturbations
    if time.time() < deadline - 5:
        current_pos = best_pos.copy()
        current_score = best_score
        
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
                
            idx = np.random.randint(N)
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = 0.01 * np.random.uniform(0.5, 1.5)
            
            new_pos = current_pos.copy()
            new_pos[idx, 0] += magnitude * np.cos(angle)
            new_pos[idx, 1] += magnitude * np.sin(angle)
            
            if np.linalg.norm(new_pos[idx]) > 1.0:
                continue
            
            new_score = _score(new_pos)
            if new_score > current_score:
                current_score = new_score
                current_pos = new_pos
                if current_score > best_score:
                    best_score = current_score
                    best_pos = new_pos.copy()
    
    return best_pos
