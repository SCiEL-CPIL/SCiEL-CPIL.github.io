"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
import scipy.linalg as la

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    def _energy_and_grad(X):
        """Compute log energy and Euclidean gradient."""
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-14)
        dist = np.sqrt(dist_sq)
        
        energy = -np.sum(np.log(dist))
        
        grad = np.zeros((N, 3))
        inv_d2 = 1.0 / dist_sq
        # Gradient contribution for i: sum_j (X_i - X_j) / d^2
        np.add.at(grad, idx_i, diff * inv_d2[:, np.newaxis])
        # Gradient contribution for j: sum_i (X_j - X_i) / d^2
        np.add.at(grad, idx_j, -diff * inv_d2[:, np.newaxis])
        
        return energy, grad

    def _project_to_sphere(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return X / norms

    def _riemannian_gradient(X, E, g_euclidean):
        """Project Euclidean gradient to tangent space of the sphere."""
        # Tangent space at X_i: vectors v such that X_i . v = 0
        # P = I - X X^T
        # g_riemannian = g_euclidean - (X . g_euclidean) X
        return g_euclidean - np.sum(X * g_euclidean, axis=1, keepdims=True) * X

    def _cayley_transform(X, G, step_size):
        """
        Retraction using Cayley transform.
        Moves X along the geodesic direction G (tangent vector).
        G must be in the tangent space.
        """
        # We want to update X to Y such that Y is on the sphere.
        # A simple and effective retraction on the sphere is the normalization, 
        # but for Riemannian optimization, we can use a more geometric step.
        # However, since we are using a generic optimizer, we will use a 
        # "geodesic step" approximation via normalization of X + step*G,
        # which is the standard exponential map approximation for small steps.
        # For better conditioning, we can use the Cayley transform if we view 
        # this as a rotation, but simple normalization is often sufficient 
        # and robust for L-BFGS. Let's stick to a robust normalized step 
        # but ensure we handle the tangent projection correctly in the optimizer loop.
        
        # Actually, standard L-BFGS works in Euclidean space. 
        # To do Riemannian L-BFGS properly, we need to map back to Euclidean 
        # coordinates or use a specific library. 
        # Since we can't easily swap out the optimizer core without risk, 
        # let's use a strategy that works well in Euclidean space but 
        # leverages the geometry for initialization and diversification.
        
        # Alternative: Use the fact that the gradient on the sphere is the 
        # projected gradient. We can run L-BFGS in R^3 but penalize 
        # deviation from the sphere strongly, or just project after every step.
        # The current code does project. 
        # Let's try a different structural change: 
        # **Symmetry Breaking / Cooperative Optimization**
        
        # This function is a placeholder for the strategy change.
        # We will implement a specific "cooperative push" in the main loop.
        pass

    def _cooperative_push(X, strength=0.01):
        """
        Identify points that are too close and push them apart.
        This helps escape local minima where points cluster.
        """
        X_new = X.copy()
        # Find pairs with small distance
        diff = X[idx_i] - X[idx_j]
        dist_sq = np.sum(diff * diff, axis=1)
        
        # Threshold for "too close"
        threshold = 0.05 # chordal distance
        close_mask = dist_sq < threshold**2
        
        if np.any(close_mask):
            close_i = idx_i[close_mask]
            close_j = idx_j[close_mask]
            
            for i, j in zip(close_i, close_j):
                d = X[i] - X[j]
                norm_d = np.linalg.norm(d)
                if norm_d < 1e-9:
                    # Random perturbation
                    d = np.random.randn(3)
                    norm_d = np.linalg.norm(d)
                
                # Push i away from j and j away from i
                # Force is proportional to 1/d^2 to be strong at close range
                force = (1.0 / (norm_d**2)) * d
                # Scale down to avoid instability
                force = force * strength * 0.1 
                
                X_new[i] += force
                X_new[j] -= force
        
        return _project_to_sphere(X_new)

    def _local_minimize(X0, max_iter=500, tol=1e-12):
        """
        Local minimization using L-BFGS-B with sphere projection.
        We add a small penalty for being off the sphere to help the optimizer,
        although projection is the primary constraint handler.
        """
        X = X0.copy()
        
        def objective_and_grad(X_flat):
            X_local = X_flat.reshape(N, 3)
            # Project to sphere to maintain constraint
            X_local = _project_to_sphere(X_local)
            e, g = _energy_and_grad(X_local)
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

    # Start with Fibonacci initialization
    best_X = _fibonacci_init()
    best_energy = _energy_and_grad(best_X)[0]
    
    # Strategy: Multi-start with Cooperative Diversification
    # 1. Generate diverse random starts
    # 2. Apply a "pre-optimization" phase of cooperative pushing to spread points
    # 3. Run local minimization
    
    rng = np.random.default_rng(42)
    n_restarts = 20
    n_pre_steps = 50 # Number of cooperative push steps before local opt
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        X_init = _random_init(restart * 1000 + 42)
        
        # Phase 1: Cooperative Diversification
        # This helps break up clusters that form in random inits
        # and provides a better starting point for the local optimizer
        X_div = X_init.copy()
        for step in range(n_pre_steps):
            if time.time() > deadline:
                break
            X_div = _cooperative_push(X_div, strength=0.05)
            # Also do a few gradient steps to settle
            # We can mix gradient descent and cooperative push
            e, g = _energy_and_grad(X_div)
            g_tan = _riemannian_gradient(X_div, e, g)
            # Simple gradient step
            lr = 0.01
            X_div = _project_to_sphere(X_div - lr * g_tan)
            
        # Phase 2: Local Minimization
        X_opt, e_opt = _local_minimize(X_div, max_iter=1000)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Final refinement on the best solution found
    # Use a higher precision local minimization
    X_final, e_final = _local_minimize(best_X, max_iter=3000, tol=1e-14)
    if e_final < best_energy:
        best_energy = e_final
        best_X = X_final
        
    # One last cooperative nudge to ensure no points are suspiciously close
    # and to polish the solution
    for _ in range(100):
        if time.time() > deadline:
            break
        best_X = _cooperative_push(best_X, strength=0.001)
        best_X, _ = _local_minimize(best_X, max_iter=100)
        
    # Ensure all points are distinct and on the unit sphere
    best_X = _project_to_sphere(best_X)
    
    # Final check for validity
    dists = np.linalg.norm(best_X[:, np.newaxis] - best_X[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Fallback to a known good configuration if we failed to separate points
        # This is a safety net, though unlikely with the cooperative push
        best_X = _fibonacci_init()
        
    return best_X
# EVOLVE-BLOCK-END
