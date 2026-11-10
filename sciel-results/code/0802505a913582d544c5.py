import numpy as np
import math

N = 150

def construct():
    # Better initialization: use a more uniform spherical distribution
    # Start with Fibonacci spiral but with better spacing
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    # Energy calculation function using math.fsum for precision
    def energy(X):
        # Compute pairwise distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        dist = np.sqrt(dist2)
        # Energy is sum over i < j of log(1 / dist) = - sum log(dist)
        # Use upper triangle only
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        # Use math.fsum for better precision
        terms = [-math.log(d[k]) for k in range(len(d))]
        e = math.fsum(terms)
        return e

    # Gradient calculation
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] # (N, N, 3)
        dist2 = np.sum(diff * diff, axis=2) # (N, N)
        np.fill_diagonal(dist2, 1.0) # Avoid div by 0
        inv_dist2 = 1.0 / dist2
        inv_dist2[np.diag_indices(N)] = 0.0
        grad = np.sum(diff * inv_dist2[..., np.newaxis], axis=1)
        return grad

    # Riemannian Gradient Descent with improved line search
    current_energy = energy(x)
    step_size = 0.05
    min_step = 1e-8
    max_iter = 5000
    no_improve_count = 0
    best_x = x.copy()
    best_e = current_energy
    prev_grad = None
    prev_step = 0.0

    for k in range(max_iter):
        g = gradient(x)
        
        # Project gradient to tangent space
        dot_gx = np.sum(g * x, axis=1)
        v = g - dot_gx[:, np.newaxis] * x
        
        # Add simple momentum-like term
        if prev_grad is not None:
            prev_dot_gx = np.sum(prev_grad * x, axis=1)
            prev_v = prev_grad - prev_dot_gx[:, np.newaxis] * x
            # Blend with previous direction for smoother convergence
            v = 0.7 * v + 0.3 * prev_v
        
        prev_grad = g
        
        # Line search: try step_size, if energy increases, shrink step_size
        accepted = False
        current_step = step_size
        for ls_iter in range(20):
            # Update
            x_new = x - current_step * v
            # Normalize to sphere
            norms = np.linalg.norm(x_new, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_new = x_new / norms
            
            # Check for invalid points (too close)
            diff_new = x_new[:, np.newaxis, :] - x_new[np.newaxis, :, :]
            dist2_new = np.sum(diff_new * diff_new, axis=2)
            np.fill_diagonal(dist2_new, 1.0)
            min_dist = np.sqrt(np.min(dist2_new))
            
            if min_dist < 1e-9:
                current_step *= 0.5
                continue
                
            e_new = energy(x_new)
            
            if e_new < current_energy:
                x = x_new
                current_energy = e_new
                step_size = current_step * 1.1 # Increase step if successful
                step_size = min(step_size, 0.2) # Cap step size
                accepted = True
                no_improve_count = 0
                if current_energy < best_e:
                    best_e = current_energy
                    best_x = x.copy()
                break
            else:
                current_step *= 0.5
                
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
            no_improve_count += 1
            if no_improve_count > 100:
                break
        
        # Early stopping if gradient is very small
        grad_norm = np.linalg.norm(v)
        if grad_norm < 1e-10:
            break

    return best_x
