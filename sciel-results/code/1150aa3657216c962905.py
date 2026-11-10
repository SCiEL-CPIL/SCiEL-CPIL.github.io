import numpy as np

N = 150

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Energy calculation function
    def energy(X):
        # Compute pairwise distances
        # X is (N, 3)
        # dist2[i, j] = ||X[i] - X[j]||^2
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        dist = np.sqrt(dist2)
        # Energy is sum over i < j of log(1 / dist) = - sum log(dist)
        # Mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        e = np.sum(np.log(1.0 / dist[mask]))
        return e

    # Gradient calculation
    # grad[i] = sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: This is -dE/dx_i? 
    # Let's re-verify.
    # E = sum_{i<j} -ln(d_ij)
    # dE/dx_i = sum_{j>i} -1/d_ij * (x_i - x_j)/d_ij + sum_{j<i} -1/d_ji * (x_j - x_i)/d_ji
    # Note term for j<i: -1/d * (x_j - x_i)/d = 1/d * (x_i - x_j)/d
    # So dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # This is the gradient of the energy. We want to minimize, so we move in direction -grad.
    
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] # (N, N, 3)
        dist2 = np.sum(diff * diff, axis=2) # (N, N)
        np.fill_diagonal(dist2, 1.0) # Avoid div by 0
        # We need (x_i - x_j) / dist^2
        # diff is x_i - x_j
        # Divide diff by dist2[..., np.newaxis]
        inv_dist2 = 1.0 / dist2
        # Set diagonal to 0 to exclude self-interaction
        inv_dist2[np.diag_indices(N)] = 0.0
        grad = np.sum(diff * inv_dist2[..., np.newaxis], axis=1)
        return grad

    # Riemannian Gradient Descent
    # We will perform a fixed number of iterations with adaptive step size.
    # Since the problem is smooth except at collisions, and we start from a good config,
    # collisions are unlikely if step size is controlled.
    
    current_energy = energy(x)
    step_size = 0.01
    min_step = 1e-6
    max_iter = 2000
    no_improve_count = 0
    best_x = x.copy()
    best_e = current_energy

    for k in range(max_iter):
        g = gradient(x)
        
        # Project gradient to tangent space
        # v = g - (g . x) x
        dot_gx = np.sum(g * x, axis=1)
        v = g - dot_gx[:, np.newaxis] * x
        
        # Line search: try step_size, if energy increases, shrink step_size
        accepted = False
        current_step = step_size
        for ls_iter in range(10):
            # Update
            x_new = x - current_step * v
            # Normalize to sphere
            norms = np.linalg.norm(x_new, axis=1, keepdims=True)
            # Handle zero norms (shouldn't happen with small steps)
            norms[norms == 0] = 1.0
            x_new = x_new / norms
            
            # Check for invalid points (too close)
            # Compute min distance
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
                step_size = current_step * 1.05 # Increase step slightly if successful
                step_size = min(step_size, 0.1) # Cap step size
                accepted = True
                no_improve_count = 0
                if current_energy < best_e:
                    best_e = current_energy
                    best_x = x.copy()
                break
            else:
                current_step *= 0.5
                
        if not accepted:
            # If no step accepted, reduce base step size
            step_size *= 0.8
            if step_size < min_step:
                break
            no_improve_count += 1
            if no_improve_count > 50:
                break

    return best_x
