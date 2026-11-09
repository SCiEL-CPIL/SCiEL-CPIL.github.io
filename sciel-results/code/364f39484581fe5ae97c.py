"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

class Optimizer:
    """Bayesian Optimizer with Gaussian Process surrogate for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Data storage for GP
        X = []
        Y = []
        
        # Hyperparameters for GP
        # Using a simple Matern 5/2 kernel
        # We will use a fixed length scale and variance for simplicity and speed,
        # or optimize them lightly if time permits.
        # Given the 5D space and 500 budget, optimizing hyperparams every step is too slow.
        # We will use a static, well-tuned kernel.
        # Length scale: The box is [-5, 5], so 10 units wide.
        # Rastrigin has local minima every 2 units (pi). So length scale ~1.0 is reasonable.
        # Signal variance: We don't know the scale of f. We will normalize Y.
        
        def matern52_kernel(X1, X2, l, sigma_f):
            # X1: (N, d), X2: (M, d)
            # Returns (N, M) kernel matrix
            # Compute squared distances
            X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
            dist_sq = np.maximum(0, X1_sq + X2_sq - 2 * X1 @ X2.T)
            dist = np.sqrt(dist_sq)
            # Matern 5/2
            sqrt5_dist = np.sqrt(5) * dist / l
            K = (1 + sqrt5_dist + 5/3 * dist_sq / l**2) * np.exp(-sqrt5_dist)
            return sigma_f**2 * K

        def gp_predict(X_new, X_train, Y_train, l, sigma_f, sigma_n):
            """
            Predict mean and variance for X_new given training data.
            X_new: (M, d)
            X_train: (N, d)
            Y_train: (N,)
            """
            K = matern52_kernel(X_train, X_train, l, sigma_f) + sigma_n**2 * np.eye(len(X_train))
            Ks = matern52_kernel(X_train, X_new, l, sigma_f)
            
            # Solve linear system
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, Ks.T)
                K_ss = matern52_kernel(X_new, X_new, l, sigma_f)
                mu = v.T @ alpha
                var = K_ss.diagonal() - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if cholesky fails
                try:
                    alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]
                    mu = Ks.T @ alpha
                    var = np.ones(len(X_new)) * np.var(Y_train) # Rough fallback
                except:
                    mu = np.zeros(len(X_new))
                    var = np.ones(len(X_new)) * (np.var(Y_train) if len(Y_train)>1 else 1.0)
            
            return mu, var

        def expected_improvement(mu, var, best_f_current):
            """
            Calculate Expected Improvement for minimization.
            mu: (M,)
            var: (M,)
            best_f_current: scalar
            """
            sd = np.sqrt(var)
            # Avoid division by zero
            sd = np.maximum(sd, 1e-10)
            
            # Z = (best_f_current - mu) / sd
            Z = (best_f_current - mu) / sd
            # EI = (best_f_current - mu) * Phi(Z) + sd * phi(Z)
            # Using standard normal CDF and PDF
            # For large Z, Phi(Z) -> 1, phi(Z) -> 0
            # For small Z, Phi(Z) -> 0.5, phi(Z) -> 0.4
            
            # Use scipy.stats.norm for accuracy
            from scipy.stats import norm
            ei = (best_f_current - mu) * norm.cdf(Z) + sd * norm.pdf(Z)
            return ei

        def acquire_next_point(X_train, Y_train, best_f_current, l, sigma_f, sigma_n):
            """
            Find the next point to evaluate by maximizing EI.
            Uses random restarts for the local optimizer of the acquisition function.
            """
            # Define the negative EI function to minimize
            def neg_ei(x):
                x = x.reshape(1, -1)
                mu, var = gp_predict(x, X_train, Y_train, l, sigma_f, sigma_n)
                ei = expected_improvement(mu[0], var[0], best_f_current)
                return -ei
            
            # Find initial best point for EI optimization
            best_neg_ei = np.inf
            best_x_acq = None
            
            # Random restarts
            for _ in range(20):
                # Sample random starting points
                x0 = rng.uniform(-5, 5, dim)
                
                # Optimize
                try:
                    res = minimize(neg_ei, x0, method='L-BFGS-B', bounds=[(-5, 5)]*dim, 
                                   options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6})
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_x_acq = res.x
                except Exception:
                    continue
            
            if best_x_acq is None:
                best_x_acq = rng.uniform(-5, 5, dim)
                
            return best_x_acq

        # Phase 1: Initial Exploration (Latin Hypercube or Random)
        # Use 20-30 points to get a good initial GP model
        n_init = 30
        if total_budget >= n_init:
            # Generate Latin Hypercube samples for better coverage
            # Simple random is also fine, but LHS is better for global coverage
            # For simplicity and speed, we use random, but stratified if possible
            # Let's use random for now, it's robust.
            X_init = rng.uniform(-5, 5, (n_init, dim))
            for i in range(n_init):
                if calls_made >= total_budget or time.time() >= deadline:
                    break
                x = X_init[i]
                f = func(x)
                X.append(x)
                Y.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
        
        # Phase 2: Bayesian Optimization Loop
        # We need to handle the scale of Y for the GP.
        # We'll normalize Y to [0, 1] or mean 0 var 1 for numerical stability.
        # Actually, since we are minimizing, and f_opt is unknown, 
        # it's safer to use raw values if the scale isn't huge, 
        # but Rastrigin can be large. 
        # Let's normalize Y to mean 0, std 1 for the GP, and convert back for EI.
        # Wait, EI depends on the scale. If we normalize, EI is in normalized units.
        # This is fine as long as we are consistent.
        
        # Hyperparameters
        # Length scale l: 1.5 (covers multiple local minima)
        # Signal variance sigma_f: 1.0 (after normalization)
        # Noise sigma_n: 0.01
        l = 1.5
        sigma_f = 1.0
        sigma_n = 0.01
        
        # We will dynamically adjust l if we get stuck, but for now fixed.
        
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
            
            if len(X) == 0:
                break
            
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Normalize Y for GP
            Y_mean = np.mean(Y_arr)
            Y_std = np.std(Y_arr)
            if Y_std < 1e-8:
                Y_std = 1.0
            Y_norm = (Y_arr - Y_mean) / Y_std
            
            # Select next point
            x_next = acquire_next_point(X_arr, Y_norm, 0.0, l, sigma_f, sigma_n) # best_f in normalized space is 0? No, best_f_norm
            # Wait, best_f_current for EI should be the best *normalized* value
            best_f_norm = (best_f - Y_mean) / Y_std
            
            x_next = acquire_next_point(X_arr, Y_norm, best_f_norm, l, sigma_f, sigma_n)
            
            # Evaluate
            f_new = func(x_next)
            X.append(x_next)
            Y.append(f_new)
            calls_made += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_next.copy()
            
            # Optional: Adaptive Hyperparameter Adjustment
            # If no improvement for 20 steps, increase length scale (more exploration)
            # This is a simple heuristic
            # For now, keep it static to ensure stability.
            
            # Check time
            if time.time() >= deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
