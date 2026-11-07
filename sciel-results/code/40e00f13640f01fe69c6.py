"""Bayesian Optimization + L-BFGS-B hybrid optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist

class Optimizer:
    """
    Optimizer using Bayesian Optimization with Gaussian Process surrogate
    and L-BFGS-B local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Clip to valid range
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        # --- Phase 1: Sobol initialization for uniform coverage ---
        # Use quasi-random points for better initial coverage in 5D
        def sobol_sequence(n_points, dim):
            """Generate Sobol-like quasi-random points in [-5, 5]^dim"""
            # Use Halton sequence as approximation to Sobol for simplicity
            points = []
            for i in range(n_points):
                point = []
                for j in range(dim):
                    # Halton sequence base j+2
                    base = j + 2
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    point.append(r)
                points.append(point)
            return np.array(points) * 10.0 - 5.0  # Scale to [-5, 5]
        
        n_init = min(20, budget // 3)  # Use up to 20 initial points
        init_points = sobol_sequence(n_init, n)
        
        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # --- Phase 2: Bayesian Optimization with GP surrogate ---
        # Collect observed points
        X_obs = [np.zeros(n)]  # Start with origin
        y_obs = [evaluate(np.zeros(n))] if y_obs is None else []
        
        # Actually, let's track all evaluations properly
        X_obs = []
        y_obs = []
        
        # Re-evaluate or track from the beginning - let's use a different approach
        # Store all evaluated points
        all_X = []
        all_y = []
        
        # We need to re-track. Let's use a wrapper that stores
        # Actually, let's just proceed with what we have from init
        # The init points were already evaluated, so let's collect them
        # This is getting complex. Let's simplify: use a fixed number of BO iterations
        
        # Simpler approach: Use BO for remaining budget
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # GP kernel: Matern 5/2 for smoothness
        def matern52_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            """Matern 5/2 kernel"""
            D = cdist(X1, X2)
            sqrt3D = np.sqrt(3.0) * D / length_scale
            K = (1.0 + sqrt3D + sqrt3D**2 / 3.0) * np.exp(-sqrt3D)
            K *= sigma_f**2
            return K
        
        # Collect initial observations from what we've evaluated
        # Since we don't have a clean record, let's do a fresh BO phase
        # with a smaller set of points
        
        # Let's restart tracking properly
        X_bo = []
        y_bo = []
        
        # Add the best point found so far
        if not np.isinf(best_f):
            X_bo.append(best_x.copy())
            y_bo.append(best_f)
        
        # Add a few random points for BO initialization
        n_bo_init = min(5, remaining_budget // 2)
        for _ in range(n_bo_init):
            x_rand = np.random.uniform(-5, 5, n)
            f_rand = evaluate(x_rand)
            if np.isfinite(f_rand):
                X_bo.append(x_rand)
                y_bo.append(f_rand)
        
        if len(X_bo) < 2:
            # Not enough data for GP, fallback to random search
            while calls < budget and time.time() < deadline:
                x_rand = np.random.uniform(-5, 5, n)
                evaluate(x_rand)
            return best_x, best_f
        
        X_bo = np.array(X_bo)
        y_bo = np.array(y_bo)
        
        # BO loop
        n_bo_candidates = 200  # Candidate points for acquisition
        bo_iterations = min(remaining_budget - 1, 30)  # Limit BO iterations
        
        for it in range(bo_iterations):
            if calls >= budget or time.time() > deadline:
                break
            
            # Fit GP
            try:
                # Standardize y for numerical stability
                y_mean = np.mean(y_bo)
                y_std = np.std(y_bo) + 1e-8
                y_norm = (y_bo - y_mean) / y_std
                
                # Fit GP parameters (simple grid search for length_scale)
                best_ls = 1.0
                best_lik = -np.inf
                
                for ls in [0.5, 1.0, 2.0, 3.0]:
                    try:
                        K = matern52_kernel(X_bo, X_bo, length_scale=ls, sigma_f=1.0)
                        K += 1e-6 * np.eye(len(X_bo))  # Jitter for numerical stability
                        
                        # Compute log marginal likelihood
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                        log_lik = -0.5 * y_norm @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y_norm) * np.log(2 * np.pi)
                        
                        if log_lik > best_lik:
                            best_lik = log_lik
                            best_ls = ls
                    except np.linalg.LinAlgError:
                        continue
                
                length_scale = best_ls
                K = matern52_kernel(X_bo, X_bo, length_scale=length_scale, sigma_f=1.0)
                K += 1e-6 * np.eye(len(X_bo))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                
                # Generate candidate points
                candidates = np.random.uniform(-5, 5, (n_bo_candidates, n))
                
                # Compute GP mean and variance at candidates
                K_s = matern52_kernel(candidates, X_bo, length_scale=length_scale, sigma_f=1.0)
                K_ss = matern52_kernel(candidates, candidates, length_scale=length_scale, sigma_f=1.0)
                
                # Mean
                mu_s = K_s @ alpha
                # Variance
                v = np.linalg.solve(L, K_s.T)
                var_s = np.diag(K_ss) - np.sum(v**2, axis=1)
                var_s = np.clip(var_s, 1e-8, None)
                std_s = np.sqrt(var_s)
                
                # Expected Improvement acquisition function
                best_y_norm = np.min(y_norm)
                Z = (best_y_norm - mu_s) / std_s
                EI = (best_y_norm - mu_s) * norm.cdf(Z) + std_s * norm.pdf(Z)
                
                # Select best candidate
                idx_best = np.argmax(EI)
                x_next = candidates[idx_best]
                
            except Exception:
                # Fallback to random if GP fails
                x_next = np.random.uniform(-5, 5, n)
            
            # Evaluate next point
            f_next = evaluate(x_next)
            
            if np.isfinite(f_next):
                X_bo = np.vstack([X_bo, x_next])
                y_bo = np.append(y_bo, f_next)
            
            # Check for convergence
            if std_s.max() < 1e-4 if 'std_s' in locals() else False:
                break
        
        # --- Phase 3: L-BFGS-B Refinement ---
        # Use remaining budget for local refinement
        remaining = budget - calls
        if remaining > 5 and not np.isinf(best_f) and time.time() < deadline:
            try:
                max_iter = max(5, remaining // 2)
                
                res = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Final random exploration if budget remains
        while calls < budget and time.time() < deadline:
            # Perturb best point
            x_perturb = best_x + np.random.randn(n) * 0.1
            x_perturb = np.clip(x_perturb, -5, 5)
            evaluate(x_perturb)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
