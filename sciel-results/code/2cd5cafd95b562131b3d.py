"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    # Vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    dist_sq_diag = np.diag(dist_sq)
    np.fill_diagonal(dist_sq, 1.0) # Temporary fix to avoid 0/0 in division, we'll mask later
    
    # Gradient term: (x_i - x_j) / |x_i - x_j|^2
    # This is diff / dist_sq
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # Note: The diagonal term was diff=0, so it contributes 0. 
    # But we replaced dist_sq diag with 1, so 0/1 = 0. Correct.
    grad = np.sum(grad_terms, axis=1)
    return grad

def _exponential_map(points, velocities, step_size):
    """Move points on the sphere using the exponential map.
    x_new = cos(step*|v|) * x + sin(step*|v|) * (v/|v|)
    """
    n = len(points)
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms < 1e-12, 1e-12, v_norms)
    
    cos_term = np.cos(step_size * v_norms_safe)
    sin_term = np.sin(step_size * v_norms_safe) / v_norms_safe
    
    # If v is 0, sin_term should be 0. 
    # With safe norm, if v_norm is tiny, sin(tiny)/tiny ~ 1. But v is 0, so 0*1=0.
    # Actually, if v is exactly 0, v_norm_safe is 1e-12. sin(1e-12*step)/1e-12 ~ step.
    # v * step = 0. So result is cos(0)*x + 0 = x. Correct.
    
    new_points = cos_term * points + sin_term * velocities
    return new_points

def construct():
    # Start with multiple diverse initial configurations and keep the best
    best_points = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Configuration 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Configuration 2: Random with minimum distance constraint
    def _random_config(seed):
        rng_local = np.random.default_rng(seed)
        pts = []
        min_dist_target = 0.15  # Target minimum chordal distance
        max_attempts = 10000
        for _ in range(N):
            for _attempt in range(max_attempts):
                p = rng_local.normal(0, 1, 3)
                p /= np.linalg.norm(p)
                # Check distance to existing points
                if pts:
                    dists = np.linalg.norm(pts - p, axis=1)
                    if np.min(dists) < min_dist_target:
                        continue
                pts.append(p)
                break
            else:
                # If we can't find a good point, just use random
                p = rng_local.normal(0, 1, 3)
                p /= np.linalg.norm(p)
                pts.append(p)
        return np.array(pts)
    
    random_points = _random_config(123)
    
    # Configuration 3: Two-cluster initialization (antipodal pairs)
    # Start with points roughly evenly distributed using spherical coordinates
    theta = np.linspace(0, np.pi, N, endpoint=False) + np.pi/N
    phi = np.linspace(0, 2*np.pi, N, endpoint=False) + np.pi/N
    # Add some randomness
    theta = theta + rng.normal(0, 0.05, N)
    phi = phi + rng.normal(0, 0.05, N)
    points3 = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    points3 /= np.linalg.norm(points3, axis=1, keepdims=True)
    
    configs = [points, random_points, points3]
    
    for init_points in configs:
        current_points = init_points.copy()
        current_energy = _log_energy(current_points)
        
        # Simulated annealing phase to escape local minima
        # Use higher temperature initially to accept uphill moves
        sa_iters = 500
        initial_temp = 0.05
        cooling_rate = 0.995
        temp = initial_temp
        
        for it in range(sa_iters):
            grad = _gradient_log_energy(current_points)
            
            # Propose a move: gradient step plus random perturbation
            step_size = 0.01 * (1 + temp * 10)
            perturbation = rng.normal(0, step_size * 0.5, current_points.shape)
            velocity = -grad * step_size + perturbation
            
            new_points = _exponential_map(current_points, velocity, 1.0)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_energy = _log_energy(new_points)
            delta_e = new_energy - current_energy
            
            # Metropolis criterion: accept if better or with probability exp(-delta/T)
            if delta_e < 0 or (temp > 1e-6 and rng.random() < np.exp(-delta_e / temp)):
                current_points = new_points
                current_energy = new_energy
            
            # Cool down
            temp *= cooling_rate
        
        # Gradient descent refinement after annealing
        lr = 0.01
        momentum = 0.9
        velocity = np.zeros_like(current_points)
        
        for it in range(3000):
            grad = _gradient_log_energy(current_points)
            
            # Update velocity with momentum
            velocity = momentum * velocity - lr * grad
            
            # Move points
            new_points = _exponential_map(current_points, velocity, 1.0)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_energy = _log_energy(new_points)
            
            # If energy increases, reject and reduce step size
            if new_energy > current_energy:
                lr *= 0.5
                velocity *= 0.5
                continue
            else:
                current_points = new_points
                current_energy = new_energy
                # Decay learning rate
                lr *= 0.9995
                
                # Check convergence
                if lr < 1e-8:
                    break
        
        # Final refinement with very small steps
        lr = 1e-8
        for it in range(200):
            grad = _gradient_log_energy(current_points)
            new_points = current_points - lr * grad
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_energy = _log_energy(new_points)
            if new_energy < current_energy:
                current_points = new_points
                current_energy = new_energy
            else:
                lr *= 0.5
                if lr < 1e-13:
                    break
        
        # Check if this configuration is better
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = current_points.copy()
    
    # Final normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # Check for close points
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < min_dist:
                min_dist = d
                
    if min_dist < 1e-9:
        # Perturb if necessary
        rng = np.random.default_rng(42)
        best_points = best_points + rng.normal(0, 1e-6, best_points.shape)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
# EVOLVE-BLOCK-END
