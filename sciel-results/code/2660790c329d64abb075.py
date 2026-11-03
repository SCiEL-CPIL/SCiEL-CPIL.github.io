import time
import numpy as np
import scipy.optimize as opt
import scipy.special as sp

N = 50
# Safety margin: 150s limit, leave 5s buffer
DEADLINE = 145.0 

def get_radius(c):
    """
    Compute the maximum radius for given centers.
    c: (N, 3) array
    """
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    # Wall distances
    # Distance to each wall: 0.5 - x, x + 0.5, etc.
    # min_i min_k (0.5 - |c_ik|) is equivalent to min over all coords of (0.5 - |coord|)
    dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    min_wall_dist = np.min(dist_to_walls)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def objective_and_grad(c_flat, beta):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes -soft_min(clearances).
    """
    c = c_flat.reshape(N, 3)
    
    # Calculate distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    d_x_pos = 0.5 - c[:, 0]
    d_x_neg = c[:, 0] + 0.5
    d_y_pos = 0.5 - c[:, 1]
    d_y_neg = c[:, 1] + 0.5
    d_z_pos = 0.5 - c[:, 2]
    d_z_neg = c[:, 2] + 0.5
    
    # Pairwise clearances (radius contribution is d/2)
    tri_idx = np.triu_indices(N, k=1)
    d_pairs = d[tri_idx] / 2.0
    
    # Wall clearances
    wall_dists = np.concatenate([
        d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg
    ])
    
    # Combine all clearances
    g_all = np.concatenate([d_pairs, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * g)))
    # Using logsumexp for stability
    log_sum_exp = sp.logsumexp(-beta * g_all)
    val = -log_sum_exp / beta
    
    # Gradient
    # weights = exp(-beta * g) / sum(exp(-beta * g))
    weights = np.exp(-beta * g_all - log_sum_exp)
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    # d(d_ij/2)/dc_i = (c_i - c_j) / d_ij * 0.5
    # d(d_ij/2)/dc_j = (c_j - c_i) / d_ij * 0.5
    # Note: The gradient of the soft min wrt g is weights.
    # So grad_c accumulates weights * d(g)/dc.
    
    for k in range(len(d_pairs)):
        i, j = tri_idx[0][k], tri_idx[1][k]
        w = weights[k]
        if w < 1e-12: 
            continue
        
        diff_ij = c[i] - c[j]
        d_ij = np.linalg.norm(diff_ij)
        if d_ij > 1e-12:
            # Factor 0.5 comes from d_pair = 0.5 * ||c_i - c_j||
            grad_c[i] += w * 0.5 * diff_ij / d_ij
            grad_c[j] -= w * 0.5 * diff_ij / d_ij
        
    # Wall part
    # Indices in g_all:
    # 0 to N-1: d_x_pos
    # N to 2N-1: d_x_neg
    # 2N to 3N-1: d_y_pos
    # 3N to 4N-1: d_y_neg
    # 4N to 5N-1: d_z_pos
    # 5N to 6N-1: d_z_neg
    
    w_start = len(d_pairs)
    
    # d(d_x_pos)/dc[:,0] = -1
    grad_c[:, 0] -= weights[w_start : w_start+N]
    # d(d_x_neg)/dc[:,0] = 1
    grad_c[:, 0] += weights[w_start+N : w_start+2*N]
    
    # d(d_y_pos)/dc[:,1] = -1
    grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
    # d(d_y_neg)/dc[:,1] = 1
    grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
    
    # d(d_z_pos)/dc[:,2] = -1
    grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
    # d(d_z_neg)/dc[:,2] = 1
    grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
    
    # We want to MINIMIZE -soft_min.
    # So return -val and -grad_c.
    return -val, -grad_c.flatten()

def _init_cubic():
    """Initial cubic grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialization inside the cube."""
    return (np.random.rand(N, 3) - 0.5)

def _simulated_annealing(centers, start_time, time_budget):
    """
    Simulated annealing to maximize the minimum clearance.
    Uses a fast score check.
    """
    best_centers = centers.copy()
    best_score = get_radius(centers)
    current_centers = centers.copy()
    current_score = best_score
    
    T = 0.02  # Initial temperature
    cooling = 0.99
    step_size = 0.05
    
    while T > 1e-5 and (time.time() - start_time) < time_budget:
        # Perturb one random center
        idx = np.random.randint(N)
        new_centers = current_centers.copy()
        new_centers[idx] += np.random.randn(3) * step_size
        # Clamp to cube
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)
        
        new_score = get_radius(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= cooling
        step_size = max(0.005, step_size * 0.99)
    
    return best_centers, best_score

def _local_refinement(centers, start_time, time_budget):
    """
    Local optimization using L-BFGS-B with explicit gradients.
    """
    best_centers = centers.copy()
    best_score = get_radius(centers)
    
    # Anneal beta from low (smooth) to high (sharp) to refine
    # Low beta -> broader basin, High beta -> closer to true min
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if (time.time() - start_time) > time_budget:
            break
            
        x0 = best_centers.ravel()
        
        try:
            # Use L-BFGS-B with explicit gradient
            # jac=True tells scipy to use the gradient returned by the function
            result = opt.minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                args=(beta,),
                jac=True,
                options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.49, 0.49)
            new_score = get_radius(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                
        except Exception:
            pass
            
    return best_centers, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Global Search with SA from a good initial guess
    centers = _init_cubic()
    best_centers = centers.copy()
    best_score = get_radius(centers)
    
    # Run SA for 30 seconds
    sa_budget = 30.0
    if time.time() - start_time < DEADLINE - 10:
        centers, score = _simulated_annealing(centers, start_time, sa_budget)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Phase 2: Local Refinement
    lr_budget = 20.0
    if time.time() - start_time < DEADLINE - 5:
        centers, score = _local_refinement(best_centers, start_time, lr_budget)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Phase 3: Additional restarts if time permits
    # Try random restarts to escape local minima
    while time.time() - start_time < DEADLINE - 15:
        # Perturb the best solution significantly
        centers = best_centers + np.random.randn(N, 3) * 0.05
        centers = np.clip(centers, -0.49, 0.49)
        
        # Short SA
        sa_short_budget = 10.0
        if time.time() - start_time < DEADLINE - 15:
            centers, score = _simulated_annealing(centers, start_time, sa_short_budget)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        
        # Short Polish
        lr_short_budget = 10.0
        if time.time() - start_time < DEADLINE - 5:
            centers, score = _local_refinement(centers, start_time, lr_short_budget)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
    # Final check
    if time.time() - start_time > DEADLINE:
        return best_centers
        
    return best_centers

if __name__ == "__main__":
    c = construct()
    print(get_radius(c))
