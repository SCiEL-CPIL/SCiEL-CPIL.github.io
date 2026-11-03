"""
Strategy: Multi-restart Riemannian L-BFGS optimization on the sphere manifold.
Instead of a single long stochastic trajectory, we perform many independent
local optimizations from diverse random starting points. Each restart uses
L-BFGS with line search in tangent-space coordinates, which converges quickly
to a local minimum. The diversity of starts increases the probability of
finding a deeper global minimum than a single trajectory can explore.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _compute_energy(X):
        """Compute total logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        # Energy = -0.5 * sum log(dist_sq)
        energy = -0.5 * np.sum(np.log(dist_sq))
        return energy
    
    def _tangent_gradient(X):
        """
        Compute the Riemannian gradient of the energy on the sphere.
        The Euclidean gradient projected to the tangent space.
        
        Energy E = sum_{i<j} log(1/|xi-xj|) = -0.5 * sum_{i<j} log(|xi-xj|^2)
        
        dE/dxi = sum_{j!=i} (xi - xj) / |xi-xj|^2
        Riemannian grad = Euclidean grad - (Euclidean grad . xi) * xi
        """
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-18)
        inv_d2 = 1.0 / dist_sq
        
        grad_euclidean = np.zeros((N, 3))
        np.add.at(grad_euclidean, idx_i, diff * inv_d2[:, np.newaxis])
        np.add.at(grad_euclidean, idx_j, -diff * inv_d2[:, np.newaxis])
        
        # Project to tangent space
        dot_grad = np.sum(grad_euclidean * X, axis=1, keepdims=True)
        grad_tangent = grad_euclidean - dot_grad * X
        
        return grad_tangent
    
    def _energy_and_grad_flat(flat):
        """
        Energy and gradient in flat tangent-space coordinates for scipy minimize.
        flat: shape (N, 3) representing tangent vectors at the current point.
        We use a retraction-based approach:
        - The variable is the tangent vector V (shape N,3)
        - The point on sphere is obtained via exponential map from a fixed reference
        Actually, for L-BFGS on manifold, we need to work in local coordinates.
        Better approach: use the fact that for small steps, we can work directly.
        
        Alternative: parameterize each point by 2 coordinates (e.g., stereographic)
        but that's complex. Instead, use the approach where we optimize in
        Cartesian coordinates with a sphere constraint handled by projection,
        or use the tangent space directly.
        
        Let's use a cleaner approach: represent each point by its tangent vector
        at a base point, and use the exponential map. But since the base point
        changes, we need to be careful.
        
        Simplest robust approach: use Cartesian coordinates with a penalty
        for deviation from the sphere, OR use scipy's constrained optimization.
        
        Actually, let's use a different strategy: optimize in Cartesian space
        with the constraint |xi|=1 handled by periodic renormalization within
        the gradient computation, using the Riemannian gradient directly.
        
        For scipy L-BFGS-B, we'll use the flat Cartesian representation and
        compute the Riemannian gradient, then use a custom step that projects
        back to the sphere.
        
        Let me use a cleaner formulation: work with the flat vector of all
        Cartesian coordinates, compute energy and Riemannian gradient, and
        use a line search with retraction.
        """
        X = flat.reshape(N, 3)
        # Normalize to ensure we're on the sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = X / norms
        
        E = _compute_energy(X)
        G = _tangent_gradient(X)
        
        return E, G.ravel()
    
    def _exp_map(X, V, step):
        """
        Exponential map on the sphere: move from X along tangent vector V
        by angle 'step'.
        X_new = cos(step) * X + sin(step) * V_hat
        where V_hat is V normalized to unit length.
        """
        V_norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norms = np.maximum(V_norms, 1e-12)
        V_unit = V / V_norms
        
        cos_step = math.cos(step)
        sin_step = math.sin(step)
        X_new = cos_step * X + sin_step * V_unit
        
        # Renormalize for numerical stability
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_new = X_new / norms
        
        return X_new
    
    def _local_optimize(X0, max_iter=500):
        """
        Perform Riemannian L-BFGS optimization starting from X0.
        Uses a custom implementation with line search in tangent space.
        """
        X = X0.copy()
        current_E = _compute_energy(X)
        
        # Store history for L-BFGS
        W = []  # list of (s, y) pairs: s = step, y = grad difference
        
        # L-BFGS parameters
        m = 20  # memory size
        
        # Initial step size
        alpha = 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Compute gradient
            G = _tangent_gradient(X)
            
            # Check convergence
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-10:
                break
            
            # Compute L-BFGS direction
            # Start with r = -G
            r = -G.copy()
            
            # L-BFGS two-loop recursion
            alphas = []
            betas = []
            for s, y in reversed(W):
                sigma = 1.0 / max(np.dot(s, s).item(), 1e-12)
                beta = sigma * np.dot(s, y).item()
                r = r + (alpha_i - beta * np.dot(s, r).item()) * s
                alphas.append(alpha_i)
                betas.append(beta)
            
            # This is getting complex. Let me use scipy.optimize.minimize with
            # a custom wrapper instead.
            break
        
        # Fall back to simple gradient descent with adaptive step size
        # This is more reliable and still effective
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            G = _tangent_gradient(X)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-10:
                break
            
            # Adaptive step size
            alpha = min(0.5 / (grad_norm + 1e-8), 0.1)
            
            # Try the step
            X_trial = _exp_map(X, G, alpha)
            E_trial = _compute_energy(X_trial)
            
            if E_trial < current_E:
                X = X_trial
                current_E = E_trial
                # Can increase alpha slightly
                alpha *= 1.1
            else:
                # Backtrack
                for _ in range(10):
                    alpha *= 0.5
                    X_trial = _exp_map(X, G, alpha)
                    E_trial = _compute_energy(X_trial)
                    if E_trial < current_E:
                        X = X_trial
                        current_E = E_trial
                        break
        
        return X, current_E
    
    def _scipy_local_optimize(X0, max_iter=200):
        """
        Use scipy L-BFGS-B with Cartesian coordinates and sphere constraint
        handled by normalization within the objective.
        """
        def objective_and_grad(flat):
            X = flat.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = X / norms
            
            E = _compute_energy(X)
            G = _tangent_gradient(X)
            
            return E, G.ravel()
        
        x0 = X0.ravel()
        
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        
        X_opt = result.x.reshape(N, 3)
        # Final normalization
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_opt = X_opt / norms
        
        E_opt = _compute_energy(X_opt)
        
        return X_opt, E_opt
    
    # Generate diverse initializations
    rng = np.random.default_rng(12345)
    
    best_X = None
    best_energy = float('inf')
    
    # First try: Fibonacci sphere (good structured start)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    X_opt, E_opt = _scipy_local_optimize(X_fib)
    if E_opt < best_energy:
        best_energy = E_opt
        best_X = X_opt.copy()
    
    # Now do random restarts
    n_restarts = 0
    while time.time() < deadline - 5.0:
        # Random initialization on sphere
        X_rand = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
        X_rand = X_rand / norms
        
        # Also try perturbed Fibonacci
        if n_restarts % 3 == 1:
            X_rand = X_fib + 0.1 * rng.standard_normal((N, 3))
            norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
            X_rand = X_rand / norms
        elif n_restarts % 3 == 2:
            # Random rotation of Fibonacci
            # Generate random rotation matrix
            Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            X_rand = X_fib @ Q.T
            # Add small perturbation
            X_rand = X_rand + 0.05 * rng.standard_normal((N, 3))
            norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
            X_rand = X_rand / norms
        
        X_opt, E_opt = _scipy_local_optimize(X_rand, max_iter=150)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
        
        n_restarts += 1
        
        # Check time
        if time.time() > deadline - 3.0:
            break
    
    # Final refinement on best solution
    if best_X is not None and time.time() < deadline - 1.0:
        X_final, E_final = _scipy_local_optimize(best_X, max_iter=300)
        if E_final < best_energy:
            best_energy = E_final
            best_X = X_final
    
    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    best_X = best_X / norms
    
    # Validate
    if not np.all(np.isfinite(best_X)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_X
