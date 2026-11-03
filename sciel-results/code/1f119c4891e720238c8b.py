import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Spectral Initialization
    def spectral_init():
        # Generate random SPD matrix
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        A = A @ A.T + np.eye(N) * 0.1
        
        # Eigendecomposition
        # We want the eigenvectors. 
        # V is N x N, columns are eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Take first 3 rows of eigvecs.T (which is eigvecs[:3, :].T)
        # This gives N points in 3D
        P = eigvecs[:3, :].T
        
        # Center
        P -= np.mean(P, axis=0)
        
        # Scale to fit in [-0.4, 0.4] to leave room for walls
        max_abs = np.max(np.abs(P))
        if max_abs > 0:
            P *= 0.4 / max_abs
            
        # Add small jitter to break symmetry
        P += np.random.normal(0, 0.01, P.shape)
        
        # Clip to ensure inside
        P = np.clip(P, -0.45, 0.45)
        
        return P

    # Energy function and gradient
    def energy_and_grad(x):
        centers = x
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-10)
        
        # Use power 4 for stability
        p = 4
        # Energy term 1: sum 1/d^p
        # d^p = (d2)^(p/2)
        # E1 = sum_{i<j} 1 / (d2_ij)^(p/2)
        
        # Gradient of 1/d^p wrt c_i
        # d/dc_i (1/d^p) = -p * (1/d^(p+2)) * (c_i - c_j)/d 
        # Wait, let's use d2.
        # E = (d2)^(-p/2)
        # dE/dc_i = (-p/2) (d2)^(-p/2 - 1) * d(d2)/dc_i
        # d(d2)/dc_i = 2 * (c_i - c_j)
        # So grad = -p * (d2)^(-(p+2)/2) * (c_i - c_j)
        
        # We want to minimize E, so we move in direction -grad.
        # Or we can just return the force (which is -grad) and let the optimizer handle it?
        # No, scipy.optimize.minimize expects the gradient of the function to be minimized.
        
        # Let's compute the gradient of the sum.
        grad = np.zeros_like(centers)
        
        # We can vectorize this.
        # coeff_{ij} = -p * (d2_ij)^(-(p+2)/2)
        # grad_i = sum_{j != i} coeff_{ij} * (c_i - c_j)
        
        # Avoid division by zero on diagonal
        d2_safe = d2.copy()
        np.fill_diagonal(d2_safe, 1e10) # Large value so coeff is ~0
        
        coeff = -p * (d2_safe ** (-(p + 2) / 2.0))
        np.fill_diagonal(coeff, 0)
        
        # grad_i = sum_j coeff_{ij} * (c_i - c_j)
        # = (sum_j coeff_{ij}) * c_i - sum_j coeff_{ij} * c_j
        
        sum_coeff = np.sum(coeff, axis=1) # Shape (N,)
        # sum_j coeff_{ij} c_j is matrix multiplication of coeff and centers
        # coeff is (N, N), centers is (N, 3). Result (N, 3).
        weighted_sum = coeff @ centers
        
        grad = sum_coeff[:, np.newaxis] * centers - weighted_sum
        
        # Wall penalty
        # E_wall = sum_{i,k} 1 / w_{ik}^p
        # w_{ik} = 0.5 - |c_{ik}|
        # This is non-differentiable at c=0. 
        # Let's use a smooth approximation: w_{ik} = 0.5 - c_{ik} for c>0, c_{ik}+0.5 for c<0.
        # Or better, use (0.5 - c_{ik})^2 + (0.5 + c_{ik})^2? No, that's always positive.
        # Let's use the distance to the nearest wall.
        # dist_i = min(0.5 - c_{ik}, 0.5 + c_{ik}) for k in 0..2
        # This is the minimum of linear functions. Non-smooth.
        
        # Alternative: Penalty for being outside the box [-0.5+r, 0.5-r]?
        # We want to keep c_ik in [-0.5, 0.5].
        # Let's use a smooth barrier: 
        # If c > 0, dist = 0.5 - c. 
        # If c < 0, dist = 0.5 + c.
        # We can approximate |c| with sqrt(c^2 + eps).
        # dist = 0.5 - sqrt(c^2 + eps)
        # This is smooth.
        
        eps = 1e-8
        abs_c = np.sqrt(centers**2 + eps)
        dist = 0.5 - abs_c
        
        # Clip dist to avoid singularity
        dist = np.maximum(dist, 1e-6)
        
        # E_wall = sum 1/dist^p
        # dE/dc = -p * dist^(-p-1) * d(dist)/dc
        # d(dist)/dc = -d(abs_c)/dc = -c / sqrt(c^2 + eps)
        
        d_abs_c_dc = centers / np.sqrt(centers**2 + eps)
        d_dist_dc = -d_abs_c_dc
        
        # grad_wall_i_k = -p * (dist_ik)^(-p-1) * d_dist_dc_ik
        coeff_wall = -p * (dist ** (-p - 1))
        grad_wall = coeff_wall * d_dist_dc
        
        # Add to total gradient
        grad += grad_wall
        
        # Calculate energy value
        # E = sum_{i<j} 1/d^p + sum_{i,k} 1/w^p
        # Sum over i<j: 0.5 * sum_{i != j} 1/d^p
        E_pairs = 0.5 * np.sum(1.0 / (d2_safe ** (p / 2.0)))
        # Diagonal is 1/1e10^(p/2) which is ~0, so 0.5 * sum all is correct for i<j
        E_walls = np.sum(1.0 / (dist ** p))
        
        return E_pairs + E_walls, grad

    # Run optimization with multiple restarts
    best_centers = None
    best_score = -np.inf
    
    for i in range(5):
        init = spectral_init()
        
        # Flatten for scipy
        x0 = init.flatten()
        
        # Define objective function for scipy
        def obj(x):
            x_reshaped = x.reshape(N, 3)
            val, grad = energy_and_grad(x_reshaped)
            return val, grad.flatten()
        
        try:
            res = minimize(obj, x0, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                centers = res.x.reshape(N, 3)
                
                # Calculate actual score (radius)
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                d2 = np.sum(diff**2, axis=2)
                np.fill_diagonal(d2, 1e10)
                d_min = np.min(np.sqrt(d2)) / 2.0
                
                w_min = np.min(0.5 - np.abs(centers))
                
                r = min(d_min, w_min)
                
                if r > best_score:
                    best_score = r
                    best_centers = centers
                    
        except Exception as e:
            pass

    # Fallback if all failed
    if best_centers is None:
        best_centers = spectral_init()
        
    return best_centers
