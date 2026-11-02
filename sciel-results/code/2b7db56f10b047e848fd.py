"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Initial Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    # Repulsion power
    k = 10.0
    
    def objective_and_gradient(x_flat):
        X = x_flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        U = X / norms
        
        # Calculate potential and gradient
        # E = sum_{i<j} (1 - U_i . U_j)^(-k)
        # Grad_i = sum_{j!=i} -k * (1 - U_i . U_j)^(-k-1) * (U_j - (U_i . U_j) U_i)
        
        # Dot products
        dots = U @ U.T
        # Set diagonal to 0 to avoid self-interaction issues in calculation, though 1-1=0 causes inf
        # We only sum i < j
        np.fill_diagonal(dots, 0.0)
        
        # Calculate terms for i < j
        # We need to be careful with the sum.
        # Let's compute the full matrix of forces.
        # Force on i from j is proportional to (1 - d_ij)^(-k-1) * (U_j - d_ij U_i)
        
        # Create a matrix of distances (1 - dots)
        dists = 1.0 - dots
        # Set diagonal to 1 to avoid division by zero in power calculation
        np.fill_diagonal(dists, 1.0)
        
        # Coefficients
        # coeff_ij = -k * dists^(-k-1)
        # We only care about i != j
        coeffs = -k * (dists ** (-k - 1.0))
        np.fill_diagonal(coeffs, 0.0)
        
        # Potential
        # E = sum_{i<j} dists_ij^(-k)
        # We can compute sum of all off-diagonal dists^(-k) and divide by 2
        pot_terms = dists ** (-k)
        np.fill_diagonal(pot_terms, 0.0)
        E = 0.5 * np.sum(pot_terms)
        
        # Gradient
        # Grad_i = sum_j coeffs_ij * (U_j - dots_ij * U_i)
        # Note: coeffs_ij is symmetric.
        # Let's compute the vector part: V = U @ coeffs.T
        # Then Grad_i = V_i - dots[i, :] dot U_i? No.
        # Grad_i = sum_j coeffs_ij * U_j - (sum_j coeffs_ij * dots_ij) * U_i
        # sum_j coeffs_ij * U_j is the i-th row of U @ coeffs.T
        # sum_j coeffs_ij * dots_ij is the i-th row of (coeffs * dots) @ ones? No, just row sum of coeffs * dots.
        
        U_coeff = U @ coeffs.T
        # coeffs * dots is element-wise product
        coeff_dots = coeffs * dots
        row_sums = np.sum(coeff_dots, axis=1)
        
        Grads = U_coeff - row_sums[:, np.newaxis] * U
        
        return E, Grads.flatten()

    # Optimize
    # L-BFGS-B is good for large scale
    res = minimize(objective_and_gradient, x0_flat, method='L-BFGS-B', jac=True, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    # Extract final points
    X_final = res.x.reshape(N, 3)
    
    # Normalize final points
    norms = np.linalg.norm(X_final, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_final = X_final / norms
    
    return X_final
# EVOLVE-BLOCK-END
