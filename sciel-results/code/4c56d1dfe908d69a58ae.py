import numpy as np
import time

N = 50
DEADLINE_SECONDS = 110

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _soft_min_energy(points, beta=50.0, eps=1e-8):
    """
    Soft-minimum objective: approximate the minimum distance using a smooth
    function that becomes sharper as beta increases. This directly targets
    maximizing the minimum distance.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists_sq = np.maximum(dists_sq, eps)
    
    # Pairwise distances (upper triangle only)
    iu = np.triu_indices_from(dists_sq, k=1)
    pair_dists = np.sqrt(dists_sq[iu])
    
    # Soft minimum of pair distances
    min_pair_soft = -np.log(np.mean(np.exp(-beta * pair_dists))) / beta
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    wall_dists = np.maximum(wall_dists, eps)
    min_wall_soft = -np.log(np.mean(np.exp(-beta * wall_dists))) / beta
    
    # The objective is to maximize min(min_pair, min_wall)
    # So minimize -min(min_pair_soft, min_wall_soft)
    # Use soft min of the two
    min_all_soft = -np.log(np.exp(-beta * min_pair_soft) + np.exp(-beta * min_wall_soft)) / beta
    
    return -min_all_soft

def _soft_min_grad(points, beta=50.0, eps=1e-8):
    """Gradient of the soft-minimum objective."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists_sq = np.maximum(dists_sq, eps)
    
    iu = np.triu_indices_from(dists_sq, k=1)
    pair_dists = np.sqrt(dists_sq[iu])
    
    # Soft min of pair distances: m_p = -log(mean(exp(-beta*d)))/beta
    # dm_p/dx_i = (beta * sum_j exp(-beta*d_ij) * (x_i - x_j)/d_ij) / (beta * sum_j exp(-beta*d_ij))
    #           = sum_j exp(-beta*d_ij) * (x_i - x_j)/d_ij / sum_j exp(-beta*d_ij)
    
    exp_neg_beta_d = np.exp(-beta * pair_dists)
    sum_exp = np.sum(exp_neg_beta_d)
    
    # For each pair (i,j), weight = exp(-beta*d_ij) / sum_exp
    weights = exp_neg_beta_d / sum_exp
    
    # Gradient contribution from pair distances
    # For point i: sum over j of weight_ij * (x_i - x_j)/d_ij
    grad_pairs = np.zeros_like(points)
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = pair_dists[idx]
        diff_ij = points[i] - points[j]
        grad_pairs[i] += w * diff_ij / d
        grad_pairs[j] -= w * diff_ij / d
    
    # Wall distances: m_w = -log(mean(exp(-beta*w)))/beta where w = 0.5 - |x|
    wall_dists = 0.5 - np.abs(points)
    wall_dists = np.maximum(wall_dists, eps)
    exp_neg_beta_w = np.exp(-beta * wall_dists)
    sum_exp_w = np.sum(exp_neg_beta_w)
    weights_w = exp_neg_beta_w / sum_exp_w
    
    # dm_w/dx_i = sum_k weight_k * d(w_k)/dx_i = sum_k weight_k * (-sign(x_ik))
    signs = np.sign(points)
    grad_walls = -np.sum(weights_w, axis=1, keepdims=True) * signs
    
    # Combine: we want to maximize min(m_p, m_w), so minimize -min(m_p, m_w)
    # The soft min of m_p and m_w: m_all = -log(exp(-beta*m_p) + exp(-beta*m_w))/beta
    # dm_all/dx = (exp(-beta*m_p)*dm_p/dx + exp(-beta*m_w)*dm_w/dx) / (exp(-beta*m_p) + exp(-beta*m_w))
    
    exp_mp = np.exp(-beta * (-np.log(sum_exp / len(iu[0])) / beta))
    exp_mw = np.exp(-beta * (-np.log(sum_exp_w / (N * 3)) / beta))
    total_exp = exp_mp + exp_mw
    
    grad = (exp_mp * grad_pairs + exp_mw * grad_walls) / total_exp
    
    return grad

def _initial_bcc(n):
    """Create a BCC (body-centered cubic) inspired initial configuration."""
    # Use a 4x4x4 grid with some perturbation
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add BCC-like offsets for some points
    bcc_offsets = np.array([
        [0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
        [0.125, -0.125, -0.125], [-0.125, 0.125, -0.125]
    ])
    
    idx = np.random.permutation(len(grid))
    selected = grid[idx[:n]].copy()
    
    # Perturb some points with BCC-like offsets
    for i in range(0, n, 4):
        if i < n:
            selected[i] += bcc_offsets[i % 4] * 0.5
    
    return np.clip(selected, -0.49, 0.49)

def _sa_optimize(points, max_time, beta_start=10.0, beta_end=100.0):
    """Simulated annealing optimization with gradient polishing."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    current_points = points.copy()
    current_energy = _soft_min_energy(current_points, beta_start)
    
    n_iter = 0
    beta = beta_start
    
    while time.time() - start_time < max_time:
        # Increase beta gradually
        progress = min(1.0, (time.time() - start_time) / max_time)
        beta = beta_start + (beta_end - beta_start) * progress**0.5
        
        # Gradient step
        grad = _soft_min_grad(current_points, beta)
        
        # Adaptive step size
        step_size = 0.01 / (1.0 + beta * 0.01)
        
        # Gradient descent step
        new_points = current_points - step_size * grad
        
        # Add random perturbation (annealing)
        temperature = 0.02 * (1.0 - progress)
        new_points += np.random.randn(N, 3) * temperature
        
        # Clip to cube
        new_points = np.clip(new_points, -0.499, 0.499)
        
        new_energy = _soft_min_energy(new_points, beta)
        
        # Accept if better or with probability exp(-delta_E / T)
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
        else:
            delta_e = new_energy - current_energy
            if temperature > 1e-6:
                accept_prob = np.exp(-delta_e / temperature)
                if np.random.random() < accept_prob:
                    current_points = new_points
                    current_energy = new_energy
        
        # Track best
        current_score = _score(current_points)
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        n_iter += 1
        
        # Every 500 iterations, do a local polish
        if n_iter % 500 == 0 and time.time() - start_time < max_time - 5:
            # Simple gradient polishing
            for _ in range(20):
                grad = _soft_min_grad(current_points, beta)
                current_points -= 0.005 * grad
                current_points = np.clip(current_points, -0.499, 0.499)
            
            polish_score = _score(current_points)
            if polish_score > best_score:
                best_score = polish_score
                best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Multiple restarts with different initializations
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
        
        if remaining_time < 15:
            break
        
        # Choose initialization
        if n_restarts == 1:
            points = _initial_bcc(N)
        elif n_restarts % 3 == 1:
            # Grid-based
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            idx = np.random.permutation(len(grid))
            points = grid[idx[:N]].copy()
            points += np.random.randn(N, 3) * 0.03
        elif n_restarts % 3 == 2:
            # Random
            points = np.random.uniform(-0.45, 0.45, (N, 3))
        else:
            # Perturbed best
            if best_points is not None:
                points = best_points + np.random.randn(N, 3) * 0.05
            else:
                points = np.random.uniform(-0.45, 0.45, (N, 3))
        
        points = np.clip(points, -0.499, 0.499)
        
        # Optimize with SA
        opt_points, opt_score = _sa_optimize(points, remaining_time)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    if best_points is None:
        best_points = _initial_bcc(N)
    
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
