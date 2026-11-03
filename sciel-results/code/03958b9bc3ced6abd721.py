"""Rewritten program: SGLD-based sphere packing optimization."""
import numpy as np
import time

N = 100

def _score(centers):
    """Compute the packing score: min of pairwise distances/2 and wall distances."""
    # Wall distances
    wall = np.min(np.min(0.5 - np.abs(centers), axis=1), axis=0)
    wall_min = np.min(wall)
    
    # Pairwise distances
    # Compute all pairwise differences
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    
    return min(wall_min, pair_min)


def _energy_and_grad(centers, r_target, beta):
    """
    Compute a smooth surrogate energy and its gradient.
    
    We use a soft-min approximation: for each constraint (pairwise or wall),
    we penalize violations of r_target using a smooth function.
    
    Energy = sum of soft penalties for constraints violated.
    """
    # Wall constraints: 0.5 - |c_ik| >= r_target for all i,k
    # Pairwise: |c_i - c_j| >= 2*r_target for all i<j
    
    # Compute wall violations
    # For each center and each axis, the distance to wall is 0.5 - |c_ik|
    # Violation if 0.5 - |c_ik| < r_target, i.e., |c_ik| > 0.5 - r_target
    
    # Smooth penalty for walls:
    # For each component c_ik, penalty = max(0, |c_ik| - (0.5 - r_target))
    # Smooth version: softplus(beta * (|c_ik| - (0.5 - r_target))) / beta
    
    # Actually, let's use a different approach. We want to maximize the minimum distance.
    # Let's use an energy that is the sum of inverse distances or something.
    
    # Alternative: use the approach from the observed improvement - repulsion-based.
    # Let's compute pairwise distances and wall distances, then use a smooth objective.
    
    # For SGLD, we need a smooth potential. Let's use:
    # E = -alpha * sum_{i<j} log(d_ij) + beta * sum_{i,k} (c_ik^2) ... no
    
    # Let me think differently. The objective is to maximize r = min(min_pair_dist/2, min_wall_dist).
    # A smooth surrogate: minimize sum of exp(-beta * d_ij / 2) for pairs + exp(-beta * wall_dist) for walls.
    # The minimum of this sum is related to the minimum distance.
    
    # Actually, let's use a simpler approach that worked: 
    # Gradient ascent on a smooth approximation of the min distance.
    
    # For pairwise: d_ij = ||c_i - c_j||
    # For wall: w_ik = 0.5 - |c_ik|
    
    # We want to maximize min(all d_ij / 2, all w_ik)
    
    # Smooth max: logsumexp(beta * values) / beta approximates max when beta is large
    # But we want to maximize the min, so let's minimize the negative.
    
    # Let's use: minimize sum exp(-beta * d_ij / 2) + sum exp(-beta * w_ik)
    # Gradient of this w.r.t. centers gives us the direction to move.
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Smooth energy: sum of exp(-beta * value) for all constraints
    # For pairs: exp(-beta * dists / 2)
    # For walls: exp(-beta * wall_dists)
    
    # We only care about the minimum, so use a small subset or all.
    # For efficiency, let's use all but with a cap.
    
    # Compute energies
    pair_energy = np.sum(np.exp(-beta * dists / 2.0))
    wall_energy = np.sum(np.exp(-beta * wall_dists))
    
    energy = pair_energy + wall_energy
    
    # Gradient
    # d/dc_i of exp(-beta * d_ij / 2) = exp(-beta * d_ij / 2) * (-beta/2) * (c_i - c_j) / d_ij
    # Sum over j != i
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # For each pair (i,j), contribution to grad_i: -beta/2 * exp(-beta * d_ij / 2) * (c_i - c_j) / d_ij
    # Contribution to grad_j: -beta/2 * exp(-beta * d_ij /2) * (c_j - c_i) / d_ij
    
    # Vectorized:
    # grad_pair_i = sum_j -beta/2 * exp(-beta * d_ij / 2) * (c_i - c_j) / d_ij
    
    # Let's compute this
    # diff[i,j] = c_i - c_j
    # d_ij = ||diff[i,j]||
    
    # factor[i,j] = -beta/2 * exp(-beta * d_ij / 2) / d_ij
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -beta / 2.0 * np.exp(-beta * dists / 2.0) / dists
    np.fill_diagonal(factor, 0.0)
    
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall gradient
    # w_ik = 0.5 - |c_ik|
    # d/dc_ik exp(-beta * w_ik) = exp(-beta * w_ik) * (-beta) * d/dc_ik (0.5 - |c_ik|)
    # d/dc_ik (0.5 - |c_ik|) = -sign(c_ik)
    # So gradient = exp(-beta * w_ik) * (-beta) * (-sign(c_ik)) = beta * exp(-beta * w_ik) * sign(c_ik)
    
    wall_grad = beta * np.exp(-beta * wall_dists) * np.sign(centers)
    # Handle zero: sign(0) = 0, which is fine (subgradient)
    
    grad = grad_pair + wall_grad
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    np.random.seed(42)
    
    # Initial position: cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # SGLD parameters
    beta = 10.0  # Softness parameter
    lr = 0.01    # Learning rate
    noise_scale = 0.01
    anneal_lr = 0.999
    anneal_noise = 0.999
    
    step = 0
    while time.time() < deadline:
        # Compute energy and gradient
        energy, grad = _energy_and_grad(centers, best_score, beta)
        
        # SGLD update
        noise = np.random.normal(0, noise_scale, size=centers.shape)
        centers = centers + lr * grad + noise
        
        # Project back to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Anneal
        lr *= anneal_lr
        noise_scale *= anneal_noise
        
        step += 1
        
        # Check score periodically
        if step % 50 == 0:
            current_score = _score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        
        # If learning rate is too small, restart with new noise
        if lr < 1e-5:
            # Restart with slight perturbation
            centers = best_centers + np.random.normal(0, 0.005, size=best_centers.shape)
            centers = np.clip(centers, -0.5, 0.5)
            lr = 0.005
            noise_scale = 0.005
            beta *= 0.8  # Reduce beta for smoother landscape
    
    # Final polish: local optimization with smaller steps
    final_lr = 0.001
    final_noise = 0.001
    for _ in range(200):
        if time.time() > deadline:
            break
        energy, grad = _energy_and_grad(centers, best_score, beta)
        noise = np.random.normal(0, final_noise, size=centers.shape)
        centers = centers + final_lr * grad + noise
        centers = np.clip(centers, -0.5, 0.5)
        
        current_score = _score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
    
    return best_centers
