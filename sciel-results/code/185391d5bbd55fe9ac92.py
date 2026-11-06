"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    def energy_and_grad(x_flat):
        """
        Computes the logarithmic energy and Euclidean gradient for L-BFGS-B.
        We work in R^3N and project the gradient to the tangent space.
        Optimized: avoid redundant sqrt, use efficient vectorization.
        """
        x = x_flat.reshape(N, 3)
        # Normalize to handle any drift
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        x = x / safe_norms
        
        # Pairwise squared distances using dot products
        dot = x @ x.T
        dist_sq = 2.0 - 2.0 * dot
        np.fill_diagonal(dist_sq, 1.0)
        
        # Gradient of energy w.r.t. x_i
        # E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
        inv_dist_sq = 1.0 / dist_sq
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        F = diff * inv_dist_sq[:, :, np.newaxis]
        grad_E = -np.sum(F, axis=1)
        
        # Project to tangent space
        dot_prod = np.sum(grad_E * x, axis=1, keepdims=True)
        grad_tangent = grad_E - dot_prod * x
        
        # Energy: E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # log(d_ij) = 0.5 * log(d_ij^2)
        # So E = -0.5 * sum_{i<j} log(d_ij) = -0.25 * sum_{i!=j} log(d_ij^2)
        log_dist_sq = np.log(dist_sq)
        np.fill_diagonal(log_dist_sq, 0.0)
        energy = -0.25 * np.sum(log_dist_sq)
        
        return energy, grad_tangent.flatten()

    def energy_only(x_flat):
        """Energy only for line search."""
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        x = x / safe_norms
        
        dot = x @ x.T
        dist_sq = 2.0 - 2.0 * dot
        np.fill_diagonal(dist_sq, 1.0)
        
        log_dist_sq = np.log(dist_sq)
        np.fill_diagonal(log_dist_sq, 0.0)
        energy = -0.25 * np.sum(log_dist_sq)
        
        return energy

    def project_to_sphere(x_flat):
        """Project points back to unit sphere."""
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        x = x / safe_norms
        return x.flatten()

    def optimize_from(x_init, max_time=20.0):
        """Run L-BFGS-B optimization from initial point."""
        x_flat = x_init.flatten()
        
        def fun(x_flat):
            return energy_only(x_flat)
        
        def jac(x_flat):
            _, g = energy_and_grad(x_flat)
            return g
        
        # Use L-BFGS-B with tight tolerances
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000}
        )
        
        x_opt = project_to_sphere(result.x)
        E_opt = energy_only(x_opt)
        
        return x_opt.reshape(N, 3), E_opt

    def polish(x_init, max_time=10.0):
        """Final high-precision polish with very tight tolerances."""
        x_flat = x_init.flatten()
        
        def fun(x_flat):
            return energy_only(x_flat)
        
        def jac(x_flat):
            _, g = energy_and_grad(x_flat)
            return g
        
        # First pass: tight tolerance
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            options={'maxiter': 15000, 'ftol': 1e-18, 'gtol': 1e-13, 'maxfun': 30000}
        )
        
        x_opt = project_to_sphere(result.x)
        E_opt = energy_only(x_opt)
        
        # Second pass: even tighter if time permits
        if time.time() < deadline - 2:
            result2 = minimize(
                fun, x_opt.flatten(), jac=jac, method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-14, 'maxfun': 40000}
            )
            x_opt2 = project_to_sphere(result2.x)
            E_opt2 = energy_only(x_opt2)
            if E_opt2 < E_opt:
                x_opt = x_opt2
                E_opt = E_opt2
        
        return x_opt.reshape(N, 3), E_opt

    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(('fib', x_fib))
    
    # 2. Perturbed Fibonacci (small noise)
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, (N, 3))
    x_pert1 = x_fib + noise
    x_pert1 = x_pert1 / np.linalg.norm(x_pert1, axis=1, keepdims=True)
    initializations.append(('pert1', x_pert1))
    
    # 3. Perturbed Fibonacci (medium noise)
    np.random.seed(999)
    noise = np.random.normal(0, 0.08, (N, 3))
    x_pert2 = x_fib + noise
    x_pert2 = x_pert2 / np.linalg.norm(x_pert2, axis=1, keepdims=True)
    initializations.append(('pert2', x_pert2))
    
    # 4. Random points
    np.random.seed(123)
    x_rand = np.random.normal(0, 1, (N, 3))
    x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
    initializations.append(('rand', x_rand))
    
    # 5. Another random with different seed
    np.random.seed(777)
    x_rand2 = np.random.normal(0, 1, (N, 3))
    x_rand2 = x_rand2 / np.linalg.norm(x_rand2, axis=1, keepdims=True)
    initializations.append(('rand2', x_rand2))
    
    # 6. Fibonacci with different golden ratio approximation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5  # phase shift
    r = np.sqrt(1.0 - z * z)
    x_fib2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(('fib2', x_fib2))
    
    # 7. Spherical code inspired: start from icosahedron-like structure and subdivide
    # Use a structured grid on sphere with better spacing
    np.random.seed(2024)
    # Start with Fibonacci, then apply a few steps of gradient descent with small step
    x_init7 = x_fib.copy()
    for _ in range(5):
        _, grad = energy_and_grad(x_init7.flatten())
        grad = grad.reshape(N, 3)
        # Project gradient to tangent
        dot_prod = np.sum(grad * x_init7, axis=1, keepdims=True)
        grad_tan = grad - dot_prod * x_init7
        # Small step
        x_init7 = x_init7 - 0.01 * grad_tan
        x_init7 = x_init7 / np.linalg.norm(x_init7, axis=1, keepdims=True)
    initializations.append(('preopt', x_init7))
    
    # 8. Random with larger perturbation
    np.random.seed(555)
    x_rand3 = np.random.normal(0, 1, (N, 3))
    x_rand3 = x_rand3 / np.linalg.norm(x_rand3, axis=1, keepdims=True)
    # Add structured perturbation
    x_rand3 = x_rand3 + 0.1 * x_fib
    x_rand3 = x_rand3 / np.linalg.norm(x_rand3, axis=1, keepdims=True)
    initializations.append(('rand3', x_rand3))
    
    # Optimize each initialization
    for name, x_init in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, E_opt = optimize_from(x_init)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            continue
    
    # Final polish on the best result - use more time for this
    if best_points is not None and time.time() < deadline - 3:
        try:
            x_opt, E_opt = polish(best_points)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            pass
    
    # If we have time, try a second polish from a slightly perturbed version
    if best_points is not None and time.time() < deadline - 2:
        try:
            np.random.seed(9999)
            x_pert = best_points + 0.005 * np.random.normal(0, 1, (N, 3))
            x_pert = x_pert / np.linalg.norm(x_pert, axis=1, keepdims=True)
            x_opt, E_opt = polish(x_pert)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_points is None:
        # Fallback to Fibonacci
        best_points = x_fib
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # Verify validity: check no points too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to clean Fibonacci
        best_points = x_fib
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
