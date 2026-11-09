"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Matern-5/2 Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        
        # Initialize with random points to cover the space
        # 10 points for 5D is sparse but enough for initial GP
        n_init = min(10, budget)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])
        
        calls_made = n_init
        
        # GP Hyperparameters (Matern 5/2)
        # Lengthscale: 1.0 is reasonable for [-5,5] box with Weierstrass oscillations
        # Signal variance: 1.0
        # Noise: 1e-8 (function is deterministic)
        ell = 1.0
        sigma_f = 1.0
        sigma_n = 1e-8
        
        best_x = X[np.argmin(Y)].copy()
        best_f = np.min(Y)
        
        # Acquisition function: Expected Improvement (EI)
        # We want to maximize EI, which means minimizing -EI
        # EI(x) = E[max(f_best - f(x), 0)]
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # 1. Fit GP (compute K and alpha)
            # In 5D with ~500 points, K is 500x500. Cholesky is O(N^3) ~ 125M ops.
            # 500 iterations * 125M = 62.5B ops. This might be too slow in Python/NumPy if not optimized.
            # NumPy uses BLAS, so 500x500 Cholesky is very fast (milliseconds).
            # 500 iterations * 1ms = 0.5s. Safe.
            
            # Compute Kernel Matrix K
            # Matern 5/2: K(x, y) = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
            # where r = ||x - y|| / ell
            
            # Vectorized distance calculation
            # X is (N, dim)
            # We need K for all pairs in X
            # diff = X[:, None, :] - X[None, :, :] -> (N, N, dim)
            # dist = np.sqrt(np.sum(diff**2, axis=-1)) -> (N, N)
            
            # To save memory and time for large N, we can use scipy.spatial.distance.cdist
            # But N is max 500, so (500, 500, 5) is 1.25M floats, 10MB. Fine.
            
            X_curr = X
            N = len(X_curr)
            
            # Calculate distances
            # Using broadcasting
            diff = X_curr[:, np.newaxis, :] - X_curr[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            
            # Matern 5/2 Kernel
            r = dist / ell
            K = (sigma_f**2) * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)
            
            # Add noise to diagonal
            K[np.diag_indices_from(K)] += sigma_n**2
            
            # Compute alpha = K^{-1} Y
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if singular
                K[np.diag_indices_from(K)] += 1e-6
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    break
            
            # Solve L v = Y
            v = np.linalg.solve(L, Y)
            # Solve L^T alpha = v
            alpha = np.linalg.solve(L.T, v)
            
            # 2. Optimize Acquisition Function
            # We use a local search (Nelder-Mead or similar) to maximize EI.
            # To be robust and fast, we use a coarse grid + local refinement for the acquisition optimizer.
            # Or simply use scipy.optimize.minimize on -EI.
            
            # Let's use a simple multi-start local search for the acquisition function
            # to avoid heavy dependency on scipy.optimize inside the loop (though it's available).
            # Actually, scipy.optimize.minimize is fine.
            
            # Define -EI function to minimize
            def neg_ei(x_new):
                # x_new is (dim,)
                x_new = x_new.reshape(1, -1)
                
                # Compute k(x_new, X) -> (1, N)
                diff_new = x_new - X_curr
                dist_new = np.sqrt(np.sum(diff_new**2, axis=-1))
                r_new = dist_new / ell
                k_new = (sigma_f**2) * (1 + np.sqrt(5)*r_new + (5/3)*r_new**2) * np.exp(-np.sqrt(5)*r_new)
                
                # Mean: mu = k_new @ alpha
                mu = np.dot(k_new, alpha)
                
                # Variance: var = K(x_new, x_new) - k_new @ K^{-1} @ k_new^T
                # K(x_new, x_new) = sigma_f^2 (since r=0)
                k_ii = sigma_f**2
                var = k_ii - np.dot(k_new, k_new.T @ alpha) # Wait, K^{-1} is L^{-T} L^{-1}
                # var = k_ii - k_new @ (L^{-1} @ (L^{-T} @ k_new^T))
                # Let's use L for this too.
                # k_new is (1, N). L is (N, N).
                # u = L^{-1} k_new^T -> solve L u = k_new.T
                k_new_T = k_new.T # (N, 1)
                u = np.linalg.solve(L, k_new_T)
                var = k_ii - np.sum(k_new * u.T)
                var = max(var, 1e-10)
                
                std = np.sqrt(var)
                
                # EI = (f_best - mu) * Phi(Z) + std * phi(Z)
                # where Z = (f_best - mu) / std
                # We want to minimize, so we return -EI
                # f_best is the current minimum.
                # Note: GP is for f(x). We want to find x where f(x) < f_best.
                # EI(x) = E[max(f_best - f(x), 0)]
                
                Z = (best_f - mu) / std
                ei = (best_f - mu) * norm.cdf(Z) + std * norm.pdf(Z)
                
                return -ei
            
            # Initial guesses for acquisition optimization
            # 1. Current best point (local refinement)
            # 2. Random points
            # 3. Centroid of data
            
            candidates = [best_x.copy()]
            for _ in range(5):
                candidates.append(rng.uniform(-5, 5, dim))
            
            best_acq_x = None
            best_acq_val = np.inf
            
            for cand in candidates:
                # Use Nelder-Mead to minimize neg_ei starting from cand
                # To keep it fast, limit maxiter
                try:
                    from scipy.optimize import minimize
                    res = minimize(neg_ei, cand, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_acq_x = res.x
                except Exception:
                    pass
            
            if best_acq_x is None:
                best_acq_x = best_x
            
            # Clip to box
            x_next = np.clip(best_acq_x, -5, 5)
            
            # 3. Evaluate function
            f_next = func(x_next)
            calls_made += 1
            
            # 4. Update GP Data
            X = np.vstack([X, x_next.reshape(1, -1)])
            Y = np.append(Y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
