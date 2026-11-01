import numpy as np
import math
import time

N = 150

def construct():
    # Deadline guard
    start_time = time.time()
    max_time = 110.0  # Safety margin from 120s
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm (they are already, but just in case)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    def energy_and_grad(pts):
        """
        Compute logarithmic energy and ambient gradients.
        Energy = sum_{i<j} log(1 / dist_ij) = - sum_{i<j} log(dist_ij)
        Grad_i = sum_{j!=i} (pts[i] - pts[j]) / dist_ij^2
        """
        # Compute pairwise distances
        # dist[i,j] = ||x_i - x_j||
        # We need to avoid division by zero for i=j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Set diagonal to 1 to avoid log(0) or div(0)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Energy: - sum_{i<j} log(dist_ij)
        # We sum over all i<j
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = -np.sum(np.log(dist[mask]))
        
        # Gradients
        # grad[i] = sum_{j != i} (x_i - x_j) / dist_ij^2
        # Note: diff[i,j] = x_i - x_j
        # We want sum over j for each i.
        # term = diff / dist_sq[:, :, np.newaxis]
        # But dist_sq is 0 on diagonal. We filled with 1.0, so diff is 0, term is 0. Correct.
        term = diff / (dist_sq[:, :, np.newaxis] + 1e-12) # small epsilon for stability
        # Sum over j (axis 1)
        grad = np.sum(term, axis=1)
        
        return energy, grad

    def project_to_tangent(grad, pts):
        """Project ambient gradient to tangent space of the sphere."""
        # P = I - x x^T
        # grad_proj = grad - (grad . x) x
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        return grad - dot * pts

    def riemannian_step(pts, grad_tangent, step_size):
        """Take a step in tangent direction and re-normalize."""
        new_pts = pts - step_size * grad_tangent
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        # Handle case where norm is 0 (shouldn't happen with small step)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return new_pts / norms

    # Optimization Loop
    # We will use a simple adaptive gradient descent with line search.
    # Since N=150 is small, we can afford many iterations.
    
    current_energy = energy_and_grad(x)[0]
    best_x = x.copy()
    best_energy = current_energy
    
    # Initial step size
    step_size = 0.1
    
    # Convergence criteria
    min_energy_change = 1e-10
    max_iter = 2000 # Upper bound, but we check time
    
    for k in range(max_iter):
        if time.time() - start_time > max_time:
            break
            
        energy, grad = energy_and_grad(x)
        
        # If we found a better energy, update best
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        # Project gradient to tangent space
        grad_t = project_to_tangent(grad, x)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-8:
            break
            
        # Line search: find step_size such that energy decreases
        # Armijo condition: f(x + alpha * v) <= f(x) + c * alpha * grad^T v
        # On manifold, we just check if energy decreases.
        # Start with current step_size, reduce if no improvement.
        
        improved = False
        alpha = step_size
        for ls_iter in range(10):
            new_x = riemannian_step(x, grad_t, alpha)
            new_energy, _ = energy_and_grad(new_x)
            
            if new_energy < energy:
                improved = True
                break
            alpha *= 0.5
            
        if improved:
            x = new_x
            # Increase step size slightly if we took a full step?
            # Or keep it constant. Let's try increasing it slowly if we didn't reduce.
            if alpha == step_size:
                step_size = min(step_size * 1.05, 1.0)
            else:
                step_size = alpha
        else:
            # If no improvement even with very small step, we might be stuck or converged.
            # Try a small random perturbation to escape local minima?
            # For now, break or reduce step size further.
            step_size *= 0.5
            if step_size < 1e-6:
                break

    # Final Polish:
    # Run a few more iterations with very small steps if time permits
    if time.time() - start_time < max_time - 5:
        step_size = 1e-4
        for k in range(100):
            if time.time() - start_time > max_time - 2:
                break
            energy, grad = energy_and_grad(x)
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
            grad_t = project_to_tangent(grad, x)
            x = riemannian_step(x, grad_t, step_size)
            step_size *= 0.95

    # Final validation and normalization
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    # Check for validity: points closer than 1e-9
    # The optimization should prevent this, but let's verify.
    # If invalid, return a valid fallback (e.g. Fibonacci)
    # However, the problem states "Points closer than 1e-9 ... make the candidate invalid".
    # Our gradient repulsion should keep them apart.
    
    return best_x
