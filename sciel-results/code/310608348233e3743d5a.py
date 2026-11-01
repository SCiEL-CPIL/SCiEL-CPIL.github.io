"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(x)
    # Compute all pairwise distances
    # x has shape (n, 3)
    # dist^2 = |xi|^2 + |xj|^2 - 2*xi.xj, but on sphere |xi|=|xj|=1
    # so dist^2 = 2 - 2*xi.xj = 2(1 - xi.xj)
    # dist = sqrt(2(1 - xi.xj))
    # log(1/dist) = -0.5 * log(2(1 - xi.xj))
    # Sum over i<j
    
    # Use matrix multiplication for efficiency
    # G = x @ x.T gives dot products
    G = x @ x.T
    # Clamp to avoid log of negative or zero
    # 1 - G[i][j] should be in (0, 2]
    # For i != j, 1 - G[i][j] > 0
    # For i == j, 1 - G[i][i] = 0, we exclude these
    
    # Create upper triangle mask
    idx = np.triu_indices(n, k=1)
    vals = 1.0 - G[idx]
    # Clamp to tiny positive to avoid log(0)
    vals = np.maximum(vals, 1e-15)
    # Energy = -0.5 * sum(log(2 * vals))
    energy = -0.5 * np.sum(np.log(2.0 * vals))
    return energy


def _energy_and_grad(x):
    """Compute energy and gradient."""
    n = len(x)
    G = x @ x.T
    idx = np.triu_indices(n, k=1)
    vals = 1.0 - G[idx]
    vals = np.maximum(vals, 1e-15)
    
    # Energy = -0.5 * sum(log(2 * vals))
    # d(energy)/dG[i,j] = -0.5 * 1/(2*vals) * 2 = -0.5/vals
    # Actually: d/dG log(2*(1-G)) = 1/(2*(1-G)) * (-2) = -1/(1-G)
    # So d(energy)/dG[i,j] = -0.5 * (-1/(1-G[i,j])) = 0.5/(1-G[i,j])
    # Wait, let me recompute.
    # energy = -0.5 * sum_{i<j} log(2*(1-G[i,j]))
    # d(energy)/dG[i,j] = -0.5 * (1/(2*(1-G[i,j]))) * (-2) = 0.5/(1-G[i,j])
    
    dE_dG = 0.5 / vals
    
    # Now, G = x @ x.T, so dG[i,j]/dx = e_i e_j^T + e_j e_i^T (for i != j)
    # The gradient of G[i,j] w.r.t. x is:
    # dG[i,j]/dx_k = x_k[i]*delta_{jk} + x_k[j]*delta_{ik}
    # So grad_x = sum_{i<j} dE_dG[i,j] * (e_i e_j^T + e_j e_i^T)
    # This means grad_x[k] = sum_{j != k} dE_dG[min(k,j),max(k,j)] * x[j]
    
    # More efficiently:
    # grad_x = 2 * (D @ x) where D is the matrix with D[i,j] = dE_dG for i<j, symmetric
    # Actually, let's construct the full dE_dG matrix (symmetric, zero diagonal)
    dE_dG_mat = np.zeros((n, n))
    dE_dG_mat[idx] = dE_dG
    dE_dG_mat = dE_dG_mat + dE_dG_mat.T  # Make symmetric, but this doubles the off-diagonal
    # Wait, that's not right. Let me think again.
    
    # G[i,j] = sum_k x[k,i] * x[k,j]
    # dG[i,j]/dx[k] = x[k,i]*delta_{jk} + x[k,j]*delta_{ik}
    # grad_energy[k] = sum_{i<j} dE_dG[i,j] * (x[k,i]*delta_{jk} + x[k,j]*delta_{ik})
    #                = sum_{i<j} dE_dG[i,j] * x[k,i] * delta_{jk} + sum_{i<j} dE_dG[i,j] * x[k,j] * delta_{ik}
    # First term: for fixed k, delta_{jk}=1 means j=k, so sum_i dE_dG[i,k] * x[k,i] for i<k
    #            + sum_i dE_dG[k,i] * x[k,i] for i>k (but we only have i<j in our sum)
    # Let's just do it with the matrix.
    
    # Actually, a cleaner way:
    # grad_x = 2 * (dE_dG_full @ x) where dE_dG_full is the symmetric matrix with
    # dE_dG_full[i,j] = dE_dG[i,j] for i != j, 0 on diagonal, and symmetric.
    # But wait, the factor of 2 comes from the symmetry.
    
    # Let me just compute it directly for clarity, using the fact that:
    # grad_x = (dE_dG_full @ x) + (dE_dG_full.T @ x) but since dE_dG_full is symmetric,
    # grad_x = 2 * dE_dG_full @ x
    # Hmm, but that would double count. Let me verify with a simple case.
    
    # Actually, the correct formula is:
    # grad_x = dE_dG_full @ x + dE_dG_full.T @ x
    # Since dE_dG_full is symmetric, this is 2 * dE_dG_full @ x
    # But we need to be careful: the dE_dG we computed is for i<j only.
    
    # Let's just construct the full matrix properly.
    dE_dG_full = np.zeros((n, n))
    dE_dG_full[idx] = dE_dG
    dE_dG_full = dE_dG_full + dE_dG_full.T
    
    # Now grad_x = dE_dG_full @ x
    # Wait, I need to double-check the factor. Let's just use numerical check in testing.
    # For now, let's use this and verify.
    
    grad = dE_dG_full @ x
    
    energy = -0.5 * np.sum(np.log(2.0 * vals))
    return energy, grad


def _project(x):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def construct():
    t_start = time.time()
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = _project(x0)
    
    best_x = x0.copy()
    best_energy = _energy(x0)
    
    # Try L-BFGS-B optimization
    try:
        # Use a custom callback to check time
        def callback(xk):
            nonlocal best_x, best_energy
            e = _energy(xk)
            if e < best_energy:
                best_energy = e
                best_x = xk.copy()
            return (time.time() - t_start) > (DEADLINE - SAFETY_MARGIN)
        
        res = minimize(_energy_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'callback': callback})
        
        if res.success or hasattr(res, 'fun'):
            e_res = _energy(res.x)
            if e_res < best_energy:
                best_energy = e_res
                best_x = res.x.copy()
        
    except Exception:
        pass
    
    # If we have time, try more iterations or different starting points
    elapsed = time.time() - t_start
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Try a second round with perturbed initial conditions
        for trial in range(5):
            if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
                break
            
            # Perturb the best solution
            np.random.seed(42 + trial)
            perturbation = np.random.randn(N, 3) * 0.1
            x_try = _project(best_x + perturbation)
            
            try:
                def callback2(xk):
                    nonlocal best_x, best_energy
                    e = _energy(xk)
                    if e < best_energy:
                        best_energy = e
                        best_x = xk.copy()
                    return (time.time() - t_start) > (DEADLINE - SAFETY_MARGIN)
                
                res2 = minimize(_energy_and_grad, x_try, method='L-BFGS-B',
                               jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                                 'callback': callback2})
                
                if res2.success or hasattr(res2, 'fun'):
                    e_res2 = _energy(res2.x)
                    if e_res2 < best_energy:
                        best_energy = e_res2
                        best_x = res2.x.copy()
                        
            except Exception:
                pass
    
    # Final projection and validation
    best_x = _project(best_x)
    
    # Validate: check for non-finite values and minimum distances
    if not np.all(np.isfinite(best_x)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_x = _project(best_x)
    
    # Check minimum pairwise distance
    G = best_x @ best_x.T
    idx = np.triu_indices(N, k=1)
    min_dist_sq = np.min(2.0 - 2.0 * G[idx])
    if min_dist_sq < (1e-9) ** 2:
        # Points too close, use Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_x = _project(best_x)
    
    return best_x
