import numpy as np
import time

N = 100

def energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient for points x on the unit sphere.
    x: (N, 3) array.
    Returns: energy (float), grad (N, 3) array.
    """
    # Compute pairwise differences
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    
    # Compute squared distances
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i == j
    # We only sum over i != j, so we can set diagonal to 1 and then zero out weights
    diag_idx = np.diag_indices(N)
    dist_sq[diag_idx] = 1.0
    
    # Weights: 1 / dist_sq
    weights = 1.0 / dist_sq
    
    # Zero out diagonal
    weights[diag_idx] = 0.0
    
    # Gradient: grad[i] = - sum_j (x[i] - x[j]) / dist_sq[i, j]
    # Note: diff[i, j] is x[i] - x[j]
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # Using dist_sq, d = sqrt(dist_sq)
    # log(d) = 0.5 * log(dist_sq)
    # We need to sum over i < j.
    # The matrix weights is symmetric. The sum of all weights * log(d) would be double counting.
    # Let's calculate the full sum and divide by 2.
    # But we must exclude i==j.
    # log_dist = 0.5 * np.log(dist_sq)
    # log_dist[diag_idx] = 0
    
    # Actually, simpler:
    # E = -0.5 * sum_{i != j} log(dist_ij)
    # dist_ij = sqrt(dist_sq_ij)
    # log(dist_ij) = 0.5 * log(dist_sq_ij)
    # So E = -0.25 * sum_{i != j} log(dist_sq_ij)
    
    # To avoid log(0) or log(negative), dist_sq is positive.
    # We set diag to 1, so log(1)=0.
    log_dist_sq = 0.5 * np.log(dist_sq)
    log_dist_sq[diag_idx] = 0.0
    
    # Sum all off-diagonal elements
    total_log_dist_sq = np.sum(log_dist_sq)
    
    # The energy is sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # sum_{i<j} log(d) = 0.5 * sum_{i!=j} log(d)
    # log(d) = 0.5 log(d^2)
    # So sum_{i<j} log(d) = 0.5 * sum_{i!=j} 0.5 log(d^2) = 0.25 * sum_{i!=j} log(d^2)
    # Energy = - 0.25 * sum_{i!=j} log(d^2)
    
    energy = -0.25 * total_log_dist_sq
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure normalized
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    current_energy, _ = energy_and_grad(x)
    best_x = x.copy()
    best_energy = current_energy
    
    step = 0.01
    min_step = 1e-12
    no_improve_count = 0
    max_no_improve = 200
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Check for stall
        if no_improve_count > max_no_improve:
            # Perturb best solution
            # Add small random noise
            noise = rng.normal(0, 0.1, x.shape)
            x_perturbed = best_x + noise
            # Normalize
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            x = x_perturbed / norms
            current_energy, _ = energy_and_grad(x)
            step = 0.01
            no_improve_count = 0
            continue
        
        # Compute gradient
        current_energy, grad = energy_and_grad(x)
        
        # Update
        x_new = x - step * grad
        
        # Project to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        # Avoid division by zero if norm is 0 (unlikely)
        norms = np.where(norms == 0, 1.0, norms)
        x_new = x_new / norms
        
        # Evaluate new energy
        new_energy, _ = energy_and_grad(x_new)
        
        if new_energy < current_energy:
            # Accept
            x = x_new
            current_energy = new_energy
            no_improve_count = 0
            # Increase step slightly
            step = min(step * 1.05, 1.0)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            # Reject
            no_improve_count += 1
            step *= 0.5
            if step < min_step:
                # Stuck, try perturbation sooner
                no_improve_count = max_no_improve

    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x
