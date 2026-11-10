"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # 1. Generate initial points using the Golden-angle (Fibonacci) spiral
    # This provides a good, non-overlapping starting configuration.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    p_cart = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Convert to spherical coordinates (theta, phi) for optimization
    # theta: polar angle [0, pi]
    # phi: azimuthal angle [0, 2*pi]
    theta = np.arccos(z)
    phi_sph = np.arctan2(p_cart[:, 1], p_cart[:, 0])
    
    # Flatten parameters for the optimizer
    x0 = np.concatenate([theta, phi_sph])
    
    # 3. Define the energy function to minimize
    # We use the Coulomb energy (Riesz s-energy with s=1) on the sphere.
    # E = sum_{i<j} 1 / (1 - p_i . p_j)
    # This potential encourages points to spread out uniformly, minimizing the covering radius.
    
    def energy(params):
        theta = params[:N]
        phi_sph = params[N:]
        
        # Convert spherical to Cartesian
        x = np.sin(theta) * np.cos(phi_sph)
        y = np.sin(theta) * np.sin(phi_sph)
        z = np.cos(theta)
        p = np.column_stack([x, y, z])
        
        # Compute dot products
        dots = p @ p.T
        
        # 1 - dot product
        one_minus_dot = 1.0 - dots
        
        # Handle the diagonal (self-interaction) to avoid division by zero
        # The diagonal elements are 0 in one_minus_dot, so 1/0 is inf.
        # We set them to 1 so that 1/1 = 1, which we will subtract later.
        np.fill_diagonal(one_minus_dot, 1.0)
        
        # Calculate the inverse distances
        inv_dist = 1.0 / one_minus_dot
        
        # Sum over all pairs i != j
        # The matrix inv_dist has 1s on the diagonal.
        # Sum of off-diagonal elements is sum(inv_dist) - N
        total_energy = 0.5 * (np.sum(inv_dist) - N)
        
        return total_energy

    # 4. Perform optimization
    # L-BFGS-B is efficient for this size and handles the smooth landscape well.
    res = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
    
    # 5. Extract the optimized parameters
    theta_opt = res.x[:N]
    phi_opt = res.x[N:]
    
    # 6. Convert back to Cartesian coordinates
    x = np.sin(theta_opt) * np.cos(phi_opt)
    y = np.sin(theta_opt) * np.sin(phi_opt)
    z = np.cos(theta_opt)
    p_opt = np.column_stack([x, y, z])
    
    # 7. Normalize vectors to ensure they lie exactly on the unit sphere
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    # Avoid division by zero if a norm is 0 (should not happen)
    norms[norms == 0] = 1.0
    p_opt = p_opt / norms
    
    return p_opt
# EVOLVE-BLOCK-END
