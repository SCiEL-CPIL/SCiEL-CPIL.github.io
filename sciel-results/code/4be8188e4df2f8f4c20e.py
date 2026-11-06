"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds


def construct():
    # 3D repulsion minimization to maximize the minimum inter-particle distance.
    # We use a Coulomb-like potential (1/d^2) which is smoother than LJ (1/d^12).
    
    best_min_dist = 0.0
    best_result = None
    
    # We will try a few different initializations:
    # 1. Jittered Grid
    # 2. Random
    
    strategies = ['grid', 'random']
    
    for strat in strategies:
        if strat == 'grid':
            # Approximate grid for N=100. 5x4x5 = 100.
            # Dimensions: 5 along x, 4 along y, 5 along z.
            # Spacing: 1/5, 1/4, 1/5.
            # Centers at -0.5 + (i+0.5)*dx
            nx, ny, nz = 5, 4, 5
            if nx * ny * nz != N:
                # Fallback to random if grid doesn't match N exactly
                # But 100 is 5*4*5, so it matches.
                pass
            
            x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
            y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
            z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
            
            positions = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            
            # Add small jitter to break symmetry and help optimizer
            np.random.seed(123)
            positions += np.random.uniform(-0.01, 0.01, size=positions.shape)
            
            # Clip to ensure inside
            positions = np.clip(positions, -0.49, 0.49)
            
        else:
            # Random start
            np.random.seed(42)
            positions = np.random.uniform(-0.45, 0.45, size=(N, 3))

        # Define the energy function: Sum of 1/d^2 for all pairs
        # This is equivalent to minimizing the sum of squared inverse distances.
        # It pushes all particles apart uniformly.
        
        def energy(pos):
            pos = pos.reshape(N, 3)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            
            # Mask diagonal
            iu = np.triu_indices(N, k=1)
            d_sq = dist_sq[iu]
            
            # Avoid division by zero
            d_sq = np.maximum(d_sq, 1e-12)
            
            # Energy = sum(1/d^2) = sum(1/d_sq)
            return np.sum(1.0 / d_sq)
            
        def gradient(pos):
            pos = pos.reshape(N, 3)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            
            # d/dx_i (1/|x_i - x_j|^2) = -2 * (x_i - x_j) / |x_i - x_j|^4
            # Sum over j != i
            # coeff[i,j] = -2 / dist_sq[i,j]^2
            # But we need to be careful with the diagonal.
            
            # Create a matrix for coefficients
            # dist_sq is (N, N)
            # We want grad[i] = sum_{j != i} -2 * diff[i,j] / dist_sq[i,j]^2
            
            # To avoid division by zero on diagonal, set dist_sq diagonal to 1
            dist_sq_safe = dist_sq.copy()
            np.fill_diagonal(dist_sq_safe, 1.0)
            
            # coeff[i,j] = -2 / dist_sq[i,j]^2
            coeff = -2.0 / (dist_sq_safe ** 2)
            np.fill_diagonal(coeff, 0.0)
            
            # grad[i] = sum_j coeff[i,j] * diff[i,j]
            grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            return grad

        bounds = Bounds(np.full(N * 3, -0.5), np.full(N * 3, 0.5))
        
        # Run optimization
        # Use L-BFGS-B
        # Since the potential is smooth, we can converge well.
        
        result = minimize(
            energy,
            positions.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        pos = result.x.reshape(N, 3)
        
        # Calculate the actual score (minimum distance / 2 and wall clearance)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist_sq = dist_sq[iu].min()
        min_dist = np.sqrt(min_dist_sq)
        
        wall_clearance = 0.5 - np.max(np.abs(pos))
        
        current_score = min(min_dist / 2.0, wall_clearance)
        
        if current_score > best_min_dist:
            best_min_dist = current_score
            best_result = pos.copy()
            
    # Final polish with a higher precision run on the best result
    if best_result is not None:
         # Re-define energy and gradient for the final polish
         # We can use a slightly higher power to sharpen the minimum distance focus
         # Or just run L-BFGS-B again with tighter tolerances.
         
         def energy_sharp(pos):
             pos = pos.reshape(N, 3)
             diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
             dist_sq = np.sum(diff ** 2, axis=2)
             iu = np.triu_indices(N, k=1)
             d_sq = dist_sq[iu]
             d_sq = np.maximum(d_sq, 1e-12)
             return np.sum(1.0 / d_sq)
             
         def gradient_sharp(pos):
             pos = pos.reshape(N, 3)
             diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
             dist_sq = np.sum(diff ** 2, axis=2)
             dist_sq_safe = dist_sq.copy()
             np.fill_diagonal(dist_sq_safe, 1.0)
             coeff = -2.0 / (dist_sq_safe ** 2)
             np.fill_diagonal(coeff, 0.0)
             grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
             return grad

         result = minimize(
             energy_sharp,
             best_result.ravel(),
             jac=gradient_sharp,
             method='L-BFGS-B',
             bounds=bounds,
             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
         )
         best_result = result.x.reshape(N, 3)
         
    # Ensure validity
    best_result = np.clip(best_result, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_result
# EVOLVE-BLOCK-END
