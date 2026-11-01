import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _energy_and_grad_ambient(X):
    """
    Compute energy and gradient for points in R^3.
    We optimize in R^3 and use a penalty term for ||x||^2 - 1, 
    or simply project back to sphere after each step in a custom optimizer.
    However, for L-BFGS-B, it's better to use a parameterization or penalty.
    
    Actually, a common trick is to optimize in R^3 and add a penalty:
    E_total = E_log + lambda * sum(||x_i||^2 - 1)^2
    
    But simpler: Use the fact that the gradient of the log energy on the sphere 
    is the projection of the ambient gradient.
    If we use L-BFGS-B in R^3, we need to handle the constraint.
    
    Alternative: Use a parameterization like quaternions or just stick to R^3 
    and project? L-BFGS-B doesn't support projection easily.
    
    Let's use a penalty method in R^3.
    E(X) = sum_{i<j} log(1/d_ij) + P * sum_i (||x_i||^2 - 1)^2
    Grad: dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j)/d_ij^2 + 4*P*(||x_i||^2-1)*x_i
    
    Wait, the current program uses Riemannian GD. The bottleneck is likely the 
    line search and the number of iterations.
    
    A more powerful approach: 
    1. Simulated Annealing to get a good global structure.
    2. Polish with L-BFGS-B on the sphere using a local chart or penalty.
    
    Let's try a different structural change: **Repulsive Potential Minimization via Force-Directed Layout with Adaptive Cooling**.
    This is effectively a specific type of SA/GD hybrid that is very robust for point distribution.
    
    Strategy:
    1. Initialize with Fibonacci.
    2. Run a high-temperature SA phase to break symmetries.
    3. Run a cooling phase (Gradient Descent with adaptive step) to polish.
    
    But the prompt asks for a MATERIAL DIFFERENCE.
    Current: Riemannian GD with backtracking.
    New: **L-BFGS-B in R^3 with penalty** + **Multi-start with diverse random seeds** + **SA for escape**.
    
    Actually, `scipy.optimize.minimize` with method='L-BFGS-B' is extremely fast per iteration compared to manual GD.
    We can define the objective in R^3 with a strong penalty for leaving the sphere.
    
    Let's implement:
    1. `objective_and_grad(X_flat)`: Computes E_log + penalty.
    2. Run L-BFGS-B from multiple starts.
    3. Use SA to perturb the best solution and re-run L-BFGS-B.
    """
    X = X_flat.reshape(N, 3)
    # Compute distances
    dists = cdist(X, X)
    np.fill_diagonal(dists, 1.0)
    d_safe = np.maximum(dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d)
    log_dists = np.log(d_safe)
    E_log = -0.5 * np.sum(log_dists)
    
    # Penalty for sphere constraint
    norms_sq = np.sum(X**2, axis=1)
    penalty_term = 1e4 * np.sum((norms_sq - 1.0)**2)
    
    E_total = E_log + penalty_term
    
    # Gradient
    # dE_log/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    inv_dists_sq = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    grad_log = -0.5 * np.sum(diff * inv_dists_sq[:, np.newaxis], axis=1)
    
    # dPenalty/dx_i = 1e4 * 2 * (||x_i||^2 - 1) * x_i
    grad_pen = 2e4 * (norms_sq - 1.0)[:, np.newaxis] * X
    
    grad_total = grad_log + grad_pen
    
    return E_total, grad_total.flatten()

def _sa_step(X, temp, rng):
    """
    Perform a step of Simulated Annealing.
    Randomly perturb points and accept if energy decreases or with probability exp(-dE/T).
    """
    X_new = X.copy()
    # Random perturbation
    noise = rng.randn(N, 3) * temp * 0.1
    X_new += noise
    # Project to sphere to maintain validity during search
    X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
    
    # Compute energies
    E_old, _ = _energy_and_grad_ambient(X.flatten())
    E_new, _ = _energy_and_grad_ambient(X_new.flatten())
    
    dE = E_new - E_old
    if dE < 0 or rng.random() < math.exp(-dE / temp):
        return X_new
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_X = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start L-BFGS-B
    # We use different initializations: Fibonacci, Random, and Spherical Codes approximations
    
    inits = []
    # Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        v = rng.randn(N, 3)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
    
    for i, X_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            X_flat = X_init.flatten()
            res = minimize(_energy_and_grad_ambient, X_flat, method='L-BFGS-B', 
                           options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50})
            
            X_opt = res.x.reshape(N, 3)
            # Re-normalize
            X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
            
            # Calculate true energy (without penalty)
            E_true, _ = _energy_and_grad_ambient(X_opt.flatten())
            # Note: _energy_and_grad_ambient includes penalty. 
            # For comparison, we should use the raw log energy.
            dists = cdist(X_opt, X_opt)
            np.fill_diagonal(dists, 1.0)
            d_safe = np.maximum(dists, 1e-12)
            E_log_only = -0.5 * np.sum(np.log(d_safe))
            
            if E_log_only < best_energy:
                best_energy = E_log_only
                best_X = X_opt.copy()
                
        except Exception:
            continue

    # Phase 2: Simulated Annealing to escape local minima
    if best_X is not None:
        X = best_X.copy()
        temp = 0.5
        cooling_rate = 0.95
        rng = np.random.RandomState(123)
        
        # Run SA for a fixed number of steps or time
        sa_start = time.time()
        while time.time() < deadline - 10.0 and temp > 0.01:
            X = _sa_step(X, temp, rng)
            temp *= cooling_rate
            
            # Periodically polish with L-BFGS-B
            if int((temp - 0.01) / 0.05) % 5 == 0: # Roughly every 5% temperature drop
                try:
                    X_flat = X.flatten()
                    res = minimize(_energy_and_grad_ambient, X_flat, method='L-BFGS-B', 
                                   options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 20})
                    X_polished = res.x.reshape(N, 3)
                    X_polished = X_polished / np.linalg.norm(X_polished, axis=1, keepdims=True)
                    
                    dists = cdist(X_polished, X_polished)
                    np.fill_diagonal(dists, 1.0)
                    d_safe = np.maximum(dists, 1e-12)
                    E_log_only = -0.5 * np.sum(np.log(d_safe))
                    
                    if E_log_only < best_energy:
                        best_energy = E_log_only
                        best_X = X_polished.copy()
                        X = best_X.copy() # Reset SA from new best
                except Exception:
                    pass

    if best_X is None:
        best_X = inits[0]
        
    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
