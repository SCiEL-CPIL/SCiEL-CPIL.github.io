"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping

def construct():
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    x0 = x0 + np.random.normal(0, 0.01, x0.shape)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

    # 2. Define Energy and Gradient
    # We use a penalty method for the local optimizer to keep it fast (L-BFGS-B)
    # Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    
    def energy_and_grad(x_flat, lam):
        x = x_flat.reshape(N, 3)
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        log_dist = 0.5 * np.log(dist_sq)
        
        triu_indices = np.triu_indices(N, 1)
        energy = -np.sum(log_dist[triu_indices])
        
        # Penalty Term
        norms_sq = np.sum(x * x, axis=1)
        penalty_val = lam * np.sum((norms_sq - 1.0) ** 2)
        
        total_energy = energy + penalty_val
        
        # Gradient Calculation
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad_energy = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        grad_penalty = 4.0 * lam * (norms_sq[:, np.newaxis] - 1.0) * x
        
        total_grad = grad_energy + grad_penalty
        
        return total_energy, total_grad.flatten()

    # 3. Local Optimizer: L-BFGS-B with high penalty
    def local_minimize(x_flat):
        # Use a high penalty to ensure points stay close to the sphere during local steps
        res = minimize(
            lambda x: energy_and_grad(x, 1000.0)[0],
            x_flat,
            jac=lambda x: energy_and_grad(x, 1000.0)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        # Normalize the result to ensure it's on the sphere for the next step
        x_opt = res.x.reshape(N, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        # Calculate the true energy on the sphere (without penalty) for comparison
        # We return the penalized energy for the optimizer, but we should track the true energy
        # However, basinhopping compares the function values returned.
        # If we return the penalized energy, the basin hopper might be confused by the penalty magnitude.
        # It is better to return the true energy, but optimize with penalty.
        # But `minimize` minimizes the penalized energy.
        # Let's return the true energy of the optimized point.
        
        # Recalculate true energy
        x_true = x_opt
        diff = x_true[:, np.newaxis, :] - x_true[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        log_dist = 0.5 * np.log(dist_sq)
        triu_indices = np.triu_indices(N, 1)
        true_energy = -np.sum(log_dist[triu_indices])
        
        return {
            'x': x_true.flatten(),
            'fun': true_energy,
            'nfev': res.nfev,
            'nit': res.nit,
            'acc': res.success
        }

    # 4. Perturbation Step: Random rotation + small jitter
    def perturbation(x):
        # Random rotation
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/4, np.pi/4)
        
        # Rotation around z-axis
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Small rotation around x-axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        R = R_x @ R_z
        x_rot = x @ R.T
        
        # Small jitter
        x_jitter = x_rot + np.random.normal(0, 0.05, x_rot.shape)
        x_jitter = x_jitter / np.linalg.norm(x_jitter, axis=1, keepdims=True)
        
        return x_jitter.flatten()

    # 5. Run Basin Hopping
    # Start from the Fibonacci spiral
    x_start = x0.flatten()
    
    # We want to minimize the true energy.
    # Basinhopping will call local_minimize, which returns the true energy.
    # It will also call perturbation.
    
    # Limit steps to ensure time limit is respected.
    # Each local minimize is ~100 L-BFGS iterations. 
    # 100 basin hopping steps * 100 L-BFGS steps * 100 points * 3 dims = manageable.
    
    res = basinhopping(
        local_minimize,
        x_start,
        stepsize=0.5,
        minimizer_kwargs={'method': 'L-BFGS-B'}, # This is ignored if local_minimize handles it, but good practice
        niter=50, # Keep low to fit in time
        seed=42
    )
    
    x_final = res.x.reshape(N, 3)
    x_final = x_final / np.linalg.norm(x_final, axis=1, keepdims=True)
    
    return x_final
# EVOLVE-BLOCK-END
