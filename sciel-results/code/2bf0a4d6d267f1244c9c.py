"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    # Use a structured initialization: icosahedron-based subdivision
    # Then refine with Riemannian conjugate gradient
    
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def calc_energy_and_grad(x):
        """Compute logarithmic energy and Riemannian gradient."""
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist[iu]))
        
        # Gradient: dE/dx_k = sum_{j != k} (1/d_kj) * (x_k - x_j)/d_kj
        # = sum_{j != k} (x_k - x_j) / d_kj^2
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad_k = sum_j inv_dist_sq[k,j] * (x_k - x_j)
        # = inv_dist_sq[k,:] @ x_k * N - inv_dist_sq[k,:] @ x
        # More carefully: grad_k = sum_j (1/d_kj^2) * (x_k - x_j)
        # = x_k * sum_j (1/d_kj^2) - sum_j (1/d_kj^2) * x_j
        
        sum_inv = np.sum(inv_dist_sq, axis=1)  # sum_j 1/d_kj^2
        weighted_sum = inv_dist_sq @ x  # sum_j (1/d_kj^2) * x_j
        
        grad = x * sum_inv[:, np.newaxis] - weighted_sum
        
        # Project to tangent space (Riemannian gradient)
        for k in range(N):
            dot = np.dot(grad[k], x[k])
            grad[k] = grad[k] - dot * x[k]
        
        return energy, grad
    
    def riemannian_cg(x, max_iter=5000, tol=1e-12):
        """Riemannian conjugate gradient with Wolfe line search."""
        energy, grad = calc_energy_and_grad(x)
        
        # Initial search direction
        d = -grad.copy()
        
        # Beta coefficient (Polak-Ribiere)
        beta = 0.0
        
        for it in range(max_iter):
            # Line search: find alpha such that Wolfe conditions hold
            # Use backtracking
            alpha = 1.0
            c1 = 1e-4
            c2 = 0.9
            
            for _ in range(50):
                # Move in direction d
                x_trial = x + alpha * d
                # Project back to sphere
                x_trial = x_trial / np.linalg.norm(x_trial, axis=1, keepdims=True)
                
                e_trial, g_trial = calc_energy_and_grad(x_trial)
                
                # Wolfe conditions
                if e_trial <= energy + c1 * alpha * np.sum(grad * d):
                    # Check curvature condition
                    if np.sum(g_trial * d) <= c2 * np.sum(grad * d):
                        break
                    elif np.sum(g_trial * d) >= 0:
                        break
                
                alpha *= 0.5
            
            x = x_trial
            energy = e_trial
            grad = g_trial
            
            # Check convergence
            if np.linalg.norm(grad) < tol:
                break
            
            # Update beta (Polak-Ribiere)
            grad_dot_d = np.sum(grad * d)
            beta_new = max(0, np.sum(grad * grad) / max(abs(grad_dot_d), 1e-16))
            
            # New search direction
            d_new = -grad + beta_new * d
            # Project to tangent space
            for k in range(N):
                dot = np.dot(d_new[k], x[k])
                d_new[k] = d_new[k] - dot * x[k]
            
            # Restart if needed
            if np.sum(d_new * d) < 0:
                d_new = -grad.copy()
                for k in range(N):
                    dot = np.dot(d_new[k], x[k])
                    d_new[k] = d_new[k] - dot * x[k]
            
            d = d_new
            beta = beta_new
        
        return x, energy
    
    # Multiple deterministic restarts
    best_x = None
    best_energy = float('inf')
    
    for restart in range(5):
        # Different initializations
        if restart == 0:
            x = fibonacci_sphere(N)
        elif restart == 1:
            # Perturbed fibonacci
            x = fibonacci_sphere(N)
            rng = np.random.RandomState(42 + restart)
            for k in range(N):
                v = rng.randn(3)
                v = v - np.dot(v, x[k]) * x[k]
                v = v / (np.linalg.norm(v) + 1e-10)
                x[k] = x[k] + 0.1 * v
                x[k] = x[k] / np.linalg.norm(x[k])
        elif restart == 2:
            # Random uniform
            rng = np.random.RandomState(123 + restart)
            x = rng.randn(N, 3)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        elif restart == 3:
            # Grid-like
            x = fibonacci_sphere(N)
            # Apply a rotation
            angle = 0.5
            axis = np.array([1, 1, 1]) / np.sqrt(3)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            x = x @ R.T
        else:
            # Another perturbation
            x = fibonacci_sphere(N)
            rng = np.random.RandomState(999 + restart)
            for k in range(N):
                v = rng.randn(3)
                v = v - np.dot(v, x[k]) * x[k]
                v = v / (np.linalg.norm(v) + 1e-10)
                x[k] = x[k] + 0.05 * v
                x[k] = x[k] / np.linalg.norm(x[k])
        
        x_opt, e_opt = riemannian_cg(x, max_iter=3000, tol=1e-10)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
    
    # Final polish with more iterations
    x_final, e_final = riemannian_cg(best_x, max_iter=5000, tol=1e-12)
    
    # Ensure unit norm
    x_final = x_final / np.linalg.norm(x_final, axis=1, keepdims=True)
    
    return x_final
# EVOLVE-BLOCK-END
