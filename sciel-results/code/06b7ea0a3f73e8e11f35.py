"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    # 1. Initialize with Fibonacci spiral
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
        # Use math.fsum for precision as per task description
        return -math.fsum(np.log(dist[iu]))

    def calc_forces(x):
        """
        Calculate repulsive forces (negative gradient of energy).
        Gradient of sum log(1/d_ij) wrt x_i is - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
        So Force = - Gradient = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, will be masked
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Force on i from j is (x_i - x_j) / dist^2
        # Sum over j for each i
        # diff[i,j] = x_i - x_j
        # force[i] = sum_j inv_dist_sq[i,j] * diff[i,j]
        forces = inv_dist_sq @ diff
        return forces

    # Phase 1: Molecular Dynamics (Velocity Verlet) with Damping
    # This helps escape local minima and settle into a good basin
    v = np.zeros_like(x)
    dt = 0.01
    damping = 0.95
    steps = 5000
    
    for step in range(steps):
        # Calculate forces
        f = calc_forces(x)
        
        # Update velocity (Verlet step 1)
        v = v * damping + 0.5 * dt * f
        
        # Update position
        x_new = x + dt * v
        
        # Project to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        x_new = x_new / norms
        
        # Calculate new forces
        f_new = calc_forces(x_new)
        
        # Update velocity (Verlet step 2)
        v = v * damping + 0.5 * dt * f_new
        
        x = x_new
        
        # Adaptive dt? Maybe reduce dt as we go
        if step % 1000 == 0:
            dt *= 0.5
            damping *= 0.99

    # Phase 2: Gradient Descent with Adaptive Step Size (Barzilai-Borwein)
    # This is a distinct optimizer from L-BFGS-B
    
    def project_tangent(grad, x):
        """Project gradient onto tangent space of the sphere."""
        # grad is (N, 3), x is (N, 3)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        return grad - dot * x

    prev_x = x.copy()
    prev_grad = None
    prev_y = None
    prev_s = None
    
    # Initial step size
    alpha = 0.01
    
    for i_iter in range(2000):
        # Calculate gradient (negative of force)
        f = calc_forces(x)
        grad = -f
        
        # Project gradient to tangent space
        grad_tan = project_tangent(grad, x)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
            
        # Barzilai-Borwein step size update
        if prev_grad is not None and prev_s is not None:
            y = grad_tan - prev_grad
            s = x - prev_x
            ys = np.sum(y * s)
            sy = np.sum(s * y)
            yy = np.sum(y * y)
            ss = np.sum(s * s)
            
            if yy > 1e-10 and ss > 1e-10:
                # B-B step size: alpha = (s^T y) / (y^T y) or (s^T s) / (s^T y)
                # We use the standard BB1: alpha = (s^T s) / (s^T y)
                if abs(ys) > 1e-10:
                    alpha_new = ss / ys
                    # Clip alpha to reasonable range
                    alpha_new = np.clip(alpha_new, 1e-4, 1.0)
                    alpha = alpha_new
        else:
            # Initial alpha estimate
            alpha = 0.01

        # Update position
        x_new = x + alpha * grad_tan
        
        # Project to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        x_new = x_new / norms
        
        # Store for next BB update
        prev_s = x_new - x
        prev_grad = grad_tan
        prev_x = x
        
        x = x_new

    # Final cleanup: ensure exact normalization
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    return x
# EVOLVE-BLOCK-END
