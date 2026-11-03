import numpy as np
import time

N = 150

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Momentum parameters
    momentum = 0.9
    v = np.zeros_like(x)
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-6
    max_step = 0.1
    
    best_x = x.copy()
    best_energy = float('inf')
    
    def compute_energy_and_grad(points):
        # points: (N, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        mask = dist_sq > 1e-12
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        
        # Gradient: sum_j (x_i - x_j) / |x_i - x_j|^2
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # log(d_ij) = 0.5 * log(dist_sq_ij)
        # E = -0.5 * sum_{i!=j} 0.5 * log(dist_sq_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
        # Note: sum_{i!=j} includes both (i,j) and (j,i), so it's 2 * sum_{i<j}
        # So E = -0.5 * sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} 0.5 log(dist_sq_ij)
        # Let's just compute the full sum and divide by 2.
        # log_d = 0.5 * log(dist_sq)
        # E = sum_{i<j} -log(d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
        
        # Using the matrix:
        # sum_{i!=j} log(dist_sq_ij) = 2 * sum_{i<j} log(dist_sq_ij)
        # So sum_{i<j} log(dist_sq_ij) = 0.5 * sum_{i!=j} log(dist_sq_ij)
        
        # We need to be careful with log(0) if points collide.
        # We used mask, so inv_dist_sq is 0 where dist_sq is small.
        # For energy, we should use the actual distances.
        
        # Let's compute energy directly from the upper triangle to be safe and simple
        # But vectorized full sum is faster.
        # log_dist_sq = np.log(dist_sq, where=mask, out=np.zeros_like(dist_sq))
        # np.log doesn't have 'where' in all versions, but np.where does.
        log_dist_sq = np.where(mask, np.log(dist_sq), 0.0)
        
        # Sum over all i != j
        total_log_dist_sq = np.sum(log_dist_sq)
        
        # E = -0.25 * total_log_dist_sq
        # Check: 
        # E = sum_{i<j} log(1/d_ij) = sum_{i<j} -log(d_ij)
        # d_ij = sqrt(dist_sq_ij)
        # log(d_ij) = 0.5 log(dist_sq_ij)
        # E = sum_{i<j} -0.5 log(dist_sq_ij)
        # total_log_dist_sq = sum_{i!=j} log(dist_sq_ij) = 2 * sum_{i<j} log(dist_sq_ij)
        # So sum_{i<j} log(dist_sq_ij) = 0.5 * total_log_dist_sq
        # E = -0.5 * (0.5 * total_log_dist_sq) = -0.25 * total_log_dist_sq
        
        energy = -0.25 * total_log_dist_sq
        return energy, grad

    # Initial energy
    current_energy, grad = compute_energy_and_grad(x)
    if current_energy < best_energy:
        best_energy = current_energy
        best_x = x.copy()

    iteration = 0
    while time.time() - start_time < time_limit:
        # Compute gradient
        energy, grad = compute_energy_and_grad(x)
        
        # Update momentum
        v = momentum * v - step_size * grad
        x = x + v
        
        # Project to sphere
        norms = np.linalg.norm(x, axis=1)
        # Handle zero norms (shouldn't happen with non-zero initial)
        norms[norms < 1e-12] = 1e-12
        x = x / norms[:, np.newaxis]
        
        # Check for collisions
        # If any distance is too small, we might have collapsed.
        # The gradient calculation handles it by ignoring pairs, but energy might be wrong.
        # We'll rely on the step size reduction if energy increases.
        
        # Adaptive step size
        if energy < best_energy:
            # Accept step, maybe increase step size slightly?
            # Let's keep it simple: if we improve, maintain step size.
            best_energy = energy
            best_x = x.copy()
            # step_size = min(max_step, step_size * 1.01)
        else:
            # Reject step, revert and reduce step size
            x = best_x.copy()
            v = 0.0 # Reset momentum on rejection to avoid oscillation
            step_size = max(min_step, step_size * 0.9)
            
        iteration += 1
        
        # If step size is very small, we might be converged
        if step_size <= min_step * 1.01:
            # Try a few more iterations with tiny steps or break
            if iteration > 10000:
                break

    return best_x

# EVOLVE-BLOCK-END
