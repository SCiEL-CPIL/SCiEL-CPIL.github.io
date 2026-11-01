"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def construct():
    # Initial guess: Fibonacci spiral with slight noise to break symmetry
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random noise to help escape degenerate symmetries
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, x0.shape)
    x = x0 + noise
    # Normalize
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    def local_energy_and_grad(x_i, x_fixed, i_idx):
        """
        Compute energy and gradient for point i given fixed other points.
        x_i: shape (3,)
        x_fixed: shape (N, 3) containing all points, but we ignore i_idx
        """
        # Normalize x_i
        x_i_norm = np.linalg.norm(x_i)
        if x_i_norm < 1e-10:
            return 0.0, np.zeros(3)
        x_i_unit = x_i / x_i_norm
        
        # Compute distances to all other points
        diff = x_fixed - x_i_unit
        dists = np.linalg.norm(diff, axis=1)
        
        # Set distance to self to 1 to avoid log(0)
        dists[i_idx] = 1.0
        
        # Energy = -sum(log(dist))
        log_dists = np.log(dists)
        energy = -np.sum(log_dists)
        
        # Gradient
        # d/dx_i [-log(||x_fixed_j - x_i||)] = (x_fixed_j - x_i) / ||x_fixed_j - x_i||^2
        inv_dists_sq = np.zeros(N)
        mask = dists > 1e-10
        inv_dists_sq[mask] = 1.0 / (dists[mask] * dists[mask])
        inv_dists_sq[i_idx] = 0.0
        
        grad = np.sum(diff * inv_dists_sq[:, np.newaxis], axis=0)
        
        # Project gradient to tangent space of sphere at x_i_unit
        dot = np.dot(grad, x_i_unit)
        grad_tangent = grad - dot * x_i_unit
        
        return energy, grad_tangent

    # Cyclic Coordinate Descent
    num_epochs = 30
    
    for epoch in range(num_epochs):
        for i in range(N):
            x_i_current = x[i].copy()
            
            # Construct orthonormal basis for tangent plane at x_i_current
            v1 = x_i_current
            v2 = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v1, v2)) > 0.9:
                v2 = np.array([0.0, 1.0, 0.0])
            u = v2 - np.dot(v2, v1) * v1
            u = u / np.linalg.norm(u)
            v = np.cross(v1, u)
            v = v / np.linalg.norm(v)
            
            def obj_2d(theta_phi):
                theta, phi = theta_phi
                ct = np.cos(theta)
                st = np.sin(theta)
                cp = np.cos(phi)
                sp = np.sin(phi)
                
                x_new = ct * v1 + st * (cp * u + sp * v)
                
                e, _ = local_energy_and_grad(x_new, x, i)
                return e
            
            def jac_2d(theta_phi):
                theta, phi = theta_phi
                ct = np.cos(theta)
                st = np.sin(theta)
                cp = np.cos(phi)
                sp = np.sin(phi)
                
                x_new = ct * v1 + st * (cp * u + sp * v)
                
                _, grad_t = local_energy_and_grad(x_new, x, i)
                
                dx_dt = -st * v1 + ct * (cp * u + sp * v)
                dx_dp = st * (-sp * u + cp * v)
                
                g_theta = np.dot(grad_t, dx_dt)
                g_phi = np.dot(grad_t, dx_dp)
                
                return np.array([g_theta, g_phi])
            
            # Optimize
            # Bounds for theta: [0, pi], phi: [-pi, pi]
            bounds = [(0, np.pi), (-np.pi, np.pi)]
            
            res = minimize(
                obj_2d, 
                [1e-6, 0.0], 
                jac=jac_2d, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            theta_opt, phi_opt = res.x
            ct = np.cos(theta_opt)
            st = np.sin(theta_opt)
            cp = np.cos(phi_opt)
            sp = np.sin(phi_opt)
            
            x[i] = ct * v1 + st * (cp * u + sp * v)
            x[i] = x[i] / np.linalg.norm(x[i])
            
    return x
# EVOLVE-BLOCK-END
