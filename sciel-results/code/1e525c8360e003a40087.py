"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def calc_energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, k=1)
        return -np.sum(np.log(dist[iu]))
        
    def calc_grad(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        # Avoid division by zero if points are very close
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # Gradient of -log(d) is -(x_i - x_j) / d^2
        # Sum over j != i
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad_i = - sum_j (x_i - x_j) / d_ij^2
        # = - ( x_i * sum_j(1/d_ij^2) - sum_j(x_j / d_ij^2) )
        sum_inv = np.sum(inv_dist_sq, axis=1)
        weighted_sum = inv_dist_sq @ x
        
        grad = - ( x * sum_inv[:, np.newaxis] - weighted_sum )
        
        # Project to tangent space
        for k in range(N):
            dot = np.dot(grad[k], x[k])
            grad[k] = grad[k] - dot * x[k]
            
        return grad

    current_energy = calc_energy(x)
    
    # Steepest Descent with Line Search
    # Parameters
    max_iter = 10000
    tol = 1e-12
    c1 = 1e-4  # Armijo condition
    rho = 0.5  # Step reduction
    
    step_size = 0.01
    
    for it in range(max_iter):
        grad = calc_grad(x)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
            
        # Direction is -grad
        direction = -grad
        
        # Line Search
        alpha = step_size
        accepted = False
        for _ in range(50):
            x_trial = x + alpha * direction
            # Normalize to sphere
            norms = np.linalg.norm(x_trial, axis=1, keepdims=True)
            x_trial = x_trial / norms
            
            e_trial = calc_energy(x_trial)
            
            if e_trial < current_energy - c1 * alpha * np.sum(grad * direction):
                accepted = True
                break
            alpha *= rho
            
        if not accepted:
            # If line search fails, try a smaller fixed step or break
            # In log energy, if we are close, gradients are large.
            # Let's just take a very small step
            alpha = 1e-5
            x_trial = x + alpha * direction
            norms = np.linalg.norm(x_trial, axis=1, keepdims=True)
            x_trial = x_trial / norms
            e_trial = calc_energy(x_trial)
            if e_trial < current_energy:
                accepted = True
            else:
                break
        
        if accepted:
            x = x_trial
            current_energy = e_trial
            # Adapt step size if we had to shrink it
            if alpha < step_size * 0.5:
                step_size = alpha
            elif alpha > step_size * 1.5:
                step_size = min(alpha * 1.5, 0.1)
        else:
            break
            
    return x
# EVOLVE-BLOCK-END
