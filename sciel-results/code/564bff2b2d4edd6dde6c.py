"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_grad_flat(x):
    """x is a flat array of length 3*N. Returns energy and gradient (flat)."""
    pts = x.reshape(N, 3)
    # Compute pairwise distances
    # Use cdist for efficiency
    D = cdist(pts, pts, metric='euclidean')
    # Avoid log(0) or log of tiny numbers; set diagonal to 1
    np.fill_diagonal(D, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # We only need upper triangle
    iu = np.triu_indices(N, 1)
    d = D[iu]
    
    # Clip to avoid log(0)
    d_safe = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d_safe))
    
    # Gradient: d/dx_i sum_{j != i} log(1/|x_i - x_j|)
    # = sum_{j != i} - (x_i - x_j) / |x_i - x_j|^2
    # Let's compute gradient efficiently
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    # D is NxN. We need for each i: sum_j (x_i - x_j) / D[i,j]^2
    # Note: D[i,j] = |x_i - x_j|
    # (x_i - x_j) is a vector.
    
    # Let's use matrix operations.
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    
    # D^2
    D2 = D * D
    np.fill_diagonal(D2, 1.0)  # Avoid division by zero
    
    # We want sum_j (x_i - x_j) / D[i,j]^2
    # This is sum_j diff[i,j] / D2[i,j]
    # Let's divide diff by D2[:, :, np.newaxis]
    inv_D2 = 1.0 / D2
    inv_D2[np.diag_indices(N)] = 0.0  # Exclude self
    
    # grad_i = - sum_j diff[i,j] * inv_D2[i,j]
    # diff is (N, N, 3), inv_D2 is (N, N)
    # We want to sum over axis 1 (j)
    # weighted_diff = diff * inv_D2[:, :, np.newaxis]
    weighted_diff = diff * inv_D2[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)  # (N, 3)
    
    return energy, grad.flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start from Fibonacci spiral
    x0 = _fibonacci_spiral(N)
    x0_flat = x0.flatten()
    
    best_x = x0_flat.copy()
    best_energy = _energy_and_grad_flat(best_x)[0]
    
    # Try multiple random perturbations and optimize
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x_start = x0_flat.copy()
        else:
            # Add random perturbation
            noise = np.random.randn(N, 3) * 0.1
            x_perturbed = x0 + noise
            # Normalize back to sphere
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            x_perturbed = x_perturbed / norms
            x_start = x_perturbed.flatten()
        
        # Use L-BFGS-B or BFGS
        # Since we're on the sphere, we can use a penalty or just project
        # Actually, L-BFGS doesn't handle constraints well.
        # Let's use a penalty method or just accept that points may drift off sphere
        # and normalize at the end. But the gradient is for the sphere manifold.
        
        # Better approach: use scipy's minimize with a custom gradient that assumes
        # the points are on the sphere, but the optimizer moves in R^3.
        # This is tricky. Let's use a different approach:
        # Use a penalty term to keep points on the sphere.
        
        def objective_with_penalty(x_flat):
            pts = x_flat.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1)
            # Penalty for not being on unit sphere
            penalty = 1000.0 * np.sum((norms - 1.0) ** 2)
            energy, grad = _energy_and_grad_flat(x_flat)
            # Gradient of penalty: 2000 * (norms - 1) * pts / norms
            penalty_grad = 2000.0 * (norms[:, np.newaxis] - 1.0) * pts / norms[:, np.newaxis]
            return energy + penalty, (grad + penalty_grad.flatten())
        
        try:
            result = minimize(
                objective_with_penalty,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_energy:
                # Normalize the result back to the sphere
                x_opt = result.x.reshape(N, 3)
                norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
                x_opt = x_opt / norms
                x_opt_flat = x_opt.flatten()
                
                # Evaluate true energy on the normalized points
                true_energy = _energy_and_grad_flat(x_opt_flat)[0]
                
                if true_energy < best_energy:
                    best_energy = true_energy
                    best_x = x_opt_flat.copy()
        
        except Exception:
            pass
    
    # Final polish with a tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            def objective_final(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1)
                penalty = 10000.0 * np.sum((norms - 1.0) ** 2)
                energy, grad = _energy_and_grad_flat(x_flat)
                penalty_grad = 20000.0 * (norms[:, np.newaxis] - 1.0) * pts / norms[:, np.newaxis]
                return energy + penalty, (grad + penalty_grad.flatten())
            
            result = minimize(
                objective_final,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result.fun < best_energy:
                x_opt = result.x.reshape(N, 3)
                norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
                x_opt = x_opt / norms
                x_opt_flat = x_opt.flatten()
                
                true_energy = _energy_and_grad_flat(x_opt_flat)[0]
                if true_energy < best_energy:
                    best_energy = true_energy
                    best_x = x_opt_flat.copy()
        
        except Exception:
            pass
    
    # Final normalization
    pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts

# EVOLVE-BLOCK-END
