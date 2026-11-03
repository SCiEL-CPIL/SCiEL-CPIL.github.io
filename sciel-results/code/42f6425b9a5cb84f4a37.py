"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.special import lpmv
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute pair indices for efficient energy/gradient calculation
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def _energy_and_grad(X):
        """Compute log energy and Euclidean gradient for points on sphere."""
        diff = X[idx_i] - X[idx_j]  # (n_pairs, 3)
        dist_sq = np.sum(diff * diff, axis=1)  # (n_pairs,)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero
        dist = np.maximum(dist, 1e-12)
        dist_sq = np.maximum(dist_sq, 1e-24)
        
        # Energy: sum log(1/d) = -sum log(d)
        energy = -np.sum(np.log(dist))
        
        # Gradient: d/dX_i [sum_{j != i} log(1/|X_i - X_j|)]
        # = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
        grad = np.zeros((N, 3))
        inv_d2 = 1.0 / dist_sq
        # grad_i += (X_i - X_j) / d^2
        np.add.at(grad, idx_i, diff * inv_d2[:, np.newaxis])
        # grad_j += (X_j - X_i) / d^2 = -diff / d^2
        np.add.at(grad, idx_j, -diff * inv_d2[:, np.newaxis])
        
        return energy, grad

    def _project_to_sphere(X):
        """Project points to the unit sphere."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return X / norms

    def _tangent_space_grad(X, euclidean_grad):
        """Project gradient to tangent space at each point."""
        # Tangent space at x is {v : x.v = 0}
        # Projection: v_t = v - (x.v)x
        dot = np.sum(X * euclidean_grad, axis=1, keepdims=True)
        return euclidean_grad - dot * X

    def _riemannian_gd(X, max_iter=500, lr=0.01, decay=0.999, tol=1e-10):
        """Riemannian Gradient Descent with exponential map."""
        X = X.copy()
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if i % 100 == 0 and i > 0:
                lr *= decay
            
            e, g = _energy_and_grad(X)
            g_tan = _tangent_space_grad(X, g)
            
            # Step size adjustment
            norm_g = np.linalg.norm(g_tan)
            if norm_g < tol:
                break
                
            # Adaptive step size
            step = lr / (1.0 + norm_g * 0.1)
            
            X_new = X - step * g_tan
            X_new = _project_to_sphere(X_new)
            
            # Check for improvement
            e_new, _ = _energy_and_grad(X_new)
            if e_new < e:
                X = X_new
            else:
                # Halve step size
                lr *= 0.5
                X_new = X - (lr / (1.0 + norm_g * 0.1)) * g_tan
                X_new = _project_to_sphere(X_new)
                e_new, _ = _energy_and_grad(X_new)
                if e_new < e:
                    X = X_new
                else:
                    break
                    
        return X

    def _local_minimize(X0, max_iter=2000, tol=1e-12):
        """L-BFGS-B local minimization with sphere projection."""
        X = X0.copy()
        
        def objective_and_grad(X_flat):
            X = X_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            e, g = _energy_and_grad(X)
            g_flat = g.reshape(-1)
            return e, g_flat
        
        result = minimize(
            objective_and_grad,
            X.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
        )
        
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        return X_opt, result.fun

    def _spherical_harmonic_init(L=12):
        """Initialize using spherical harmonic density field."""
        # Generate random spherical harmonic coefficients
        rng = np.random.default_rng(42)
        
        # Create a grid of points on sphere for evaluation
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        
        # Compute spherical harmonics Y_l^m
        def Y_lm(l, m, theta, phi):
            """Compute spherical harmonic Y_l^m."""
            # Associated Legendre polynomial
            P_lm = lpmv(abs(m), l, np.cos(theta))
            # Condon-Shortley phase
            phase = (-1)**m
            # Normalization
            norm = np.sqrt((2*l + 1) / (4*np.pi) * 
                          math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
            # Complex exponential
            exp_term = np.exp(1j * m * phi)
            return phase * norm * P_lm * exp_term
        
        # Build density field from random coefficients
        coeffs = {}
        for l in range(L + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    coeffs[(l, m)] = rng.standard_normal() + 0j
                else:
                    coeffs[(l, m)] = rng.standard_normal() + 1j * rng.standard_normal()
        
        # Evaluate density field
        density = np.zeros_like(Theta, dtype=complex)
        for (l, m), c in coeffs.items():
            if l == 0 and m == 0:
                continue  # Skip constant term
            Y = Y_lm(l, m, Theta, Phi)
            density += c * Y
        
        # Take real part and normalize to positive
        density_real = np.real(density)
        density_real = density_real - np.min(density_real)
        density_real = density_real + 1e-10
        
        # Sample points from this density using importance sampling
        # Flatten and normalize
        flat_density = density_real.flatten()
        probs = flat_density / np.sum(flat_density)
        
        # Sample N points
        indices = rng.choice(len(flat_density), size=N, replace=False, p=probs)
        
        # Convert back to theta, phi
        theta_idx = indices // n_phi
        phi_idx = indices % n_phi
        theta_sampled = Theta[theta_idx, phi_idx]
        phi_sampled = Phi[theta_idx, phi_idx]
        
        # Convert to Cartesian
        x = np.sin(theta_sampled) * np.cos(phi_sampled)
        y = np.sin(theta_sampled) * np.sin(phi_sampled)
        z = np.cos(theta_sampled)
        
        return np.column_stack([x, y, z])

    def _fibonacci_init():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def _random_init(seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    # Phase 1: Generate diverse initializations
    best_X = _fibonacci_init()
    best_energy = _energy_and_grad(best_X)[0]
    
    # Try spherical harmonic initialization
    X_sh = _spherical_harmonic_init(L=12)
    X_sh = _project_to_sphere(X_sh)
    e_sh, _ = _energy_and_grad(X_sh)
    if e_sh < best_energy:
        best_energy = e_sh
        best_X = X_sh
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        X_init = _random_init(restart * 1000 + 42)
        
        # First do a quick Riemannian GD to get close to a local min
        X_rg = _riemannian_gd(X_init, max_iter=200, lr=0.05)
        
        # Then refine with L-BFGS-B
        X_opt, e_opt = _local_minimize(X_rg, max_iter=1000, tol=1e-10)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Phase 2: Final refinement with Riemannian GD + L-BFGS-B
    if time.time() < deadline - 10:
        # Riemannian GD polish
        X_rg = _riemannian_gd(best_X, max_iter=300, lr=0.01)
        e_rg, _ = _energy_and_grad(X_rg)
        
        if e_rg < best_energy:
            best_energy = e_rg
            best_X = X_rg
        
        # L-BFGS-B final polish
        X_final, e_final = _local_minimize(best_X, max_iter=3000, tol=1e-14)
        if e_final < best_energy:
            best_energy = e_final
            best_X = X_final
    
    # Ensure all points are distinct and on the unit sphere
    best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
