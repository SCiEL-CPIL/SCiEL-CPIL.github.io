import numpy as np
import time
import math
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Initial Exploration: Low-discrepancy sequence (Sobol-like via Halton for simplicity/robustness)
        # Halton sequence is deterministic and space-filling.
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def generate_halton_points(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(dim):
                    # Use the first 'dim' prime numbers
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = primes[j]
                    point.append(halton_sequence(i, base))
                points.append(point)
            return np.array(points)

        # Map [0,1]^dim to [-4.5, 4.5]^dim to stay within bounds and allow margin
        n_init = 40
        init_points = generate_halton_points(n_init, dim)
        init_points = init_points * 9.0 - 4.5
        
        for x in init_points:
            if n_calls >= budget or time.time() > deadline:
                break
            timed_call(x)

        if best_x is None:
            best_x = np.zeros(dim)
            timed_call(best_x)

        # 2. Surrogate Model (Kriging/Gaussian Process)
        # We implement a simple GP with Matern kernel (nu=2.5) for smoothness.
        # This is better for continuous optimization than simple RBF sometimes, 
        # but RBF is safer for numerical stability. Let's use RBF (Matern infinity).
        
        X_train = [best_x.copy()]
        y_train = [best_f]
        
        # Add the initial points to the training set
        # We need to re-evaluate or store them? 
        # We didn't store all initial f values in a list corresponding to X.
        # Let's re-do the init loop to store them properly.
        
        # RE-INIT: Store points
        X_init = []
        y_init = []
        # Reset state? No, we already made calls. 
        # We need to keep track of all calls for the GP.
        # The previous loop called timed_call but didn't store X, y.
        # This is a problem for building the GP.
        # Let's restart the logic or just use the best point and refine?
        # Better: Implement the GP loop starting now, and we will use the best_x found so far.
        # But a GP with 1 point is useless.
        # We should have collected the initial points.
        
        # Let's fix the initialization to collect data.
        # Since we already used ~40 calls, let's just assume we have best_x.
        # To make the GP work, we need a dataset. 
        # If we didn't store the first 40, we can't use them for the GP directly without re-evaluating (wastes budget).
        # Alternative: Use the Halton points as the initial dataset for the GP, 
        # but we need their f-values. 
        # Let's restart the call counting logic in our head: 
        # The previous block consumed calls. We don't have the y-values.
        # This is a flaw in the quick draft. 
        # Solution: Collect points in the first phase.
        
        # Let's restart the __call__ body logic properly.
        # The code above already executed? No, this is the source code generation.
        # I will write the full correct code below.

        # CORRECTED INITIALIZATION PHASE
        # We will clear the previous partial logic and write it cleanly.
        
        # (The code below replaces the previous init section)
        
        # --- Start of Clean Logic ---
        
        # 1. Halton Init
        X_data = []
        y_data = []
        
        for x in init_points:
            if n_calls >= budget or time.time() > deadline:
                break
            val = timed_call(x)
            X_data.append(x)
            y_data.append(val)
            
        if not X_data:
            x0 = np.zeros(dim)
            val0 = timed_call(x0)
            X_data.append(x0)
            y_data.append(val0)
            
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 2. Bayesian Optimization Loop with GP
        # Kernel: Matern 5/2 or RBF. RBF is smooth. Matern 5/2 is less smooth.
        # For BBOB f18 (Schaffers F7), it's ill-conditioned but continuous.
        # Let's use a simple RBF kernel with automatic relevance determination (ARD)
        # or just a single length scale for simplicity and speed.
        
        def compute_kernel(X1, X2, length_scale=1.0, noise=1e-8):
            # RBF Kernel
            sq_dist = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-0.5 * sq_dist / (length_scale ** 2))
        
        # We will use a fixed length scale initially, or adapt it.
        # Adapting hyperparameters is expensive. 
        # Let's use a fixed length scale of 1.0, which is reasonable for [-5,5]^5.
        # Actually, Schaffers F7 has a specific structure. 
        # Let's use length_scale = 1.0.
        
        length_scale = 1.0
        noise = 1e-8
        
        # Acquire Function: Expected Improvement (EI)
        # EI(x) = (mu(x) - y_best - xi) * Phi((mu(x) - y_best - xi) / sigma(x)) + sigma(x) * phi((mu(x) - y_best - xi) / sigma(x))
        # where mu, sigma are GP mean and std.
        # xi is exploration-exploitation tradeoff.
        
        def expected_improvement(mu, sigma, y_best, xi=0.01):
            z = (mu - y_best - xi) / (sigma + 1e-8)
            return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Main Loop
        # We will alternate between:
        # 1. GP-based step (1 call)
        # 2. Local Refinement (Nelder-Mead or CMA-ES) if improvement is found
        
        stagnation_counter = 0
        max_stagnation = 10
        
        while n_calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a local optimizer run
            # If we are stuck, use local optimizer.
            
            # Update GP
            # We need to solve for GP parameters.
            # K = compute_kernel(X_data, X_data, length_scale) + noise * I
            # alpha = K^-1 y
            # mean(x) = K(x, X) @ alpha
            # var(x) = K(x, x) - K(x, X) @ K^-1 @ K(X, x)
            
            try:
                K = compute_kernel(X_data, X_data, length_scale) + noise * np.eye(len(X_data))
                # Regularize
                K += 1e-6 * np.eye(len(X_data))
                
                # Cholesky decomposition for stability
                try:
                    L = np.linalg.cholesky(K)
                    # Solve L @ v = y
                    v = np.linalg.solve(L, y_data)
                    # alpha = L^-T @ v
                    alpha = np.linalg.solve(L.T, v)
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Fallback to pinv
                    K_inv = np.linalg.pinv(K)
                    alpha = K_inv @ y_data
                    L_inv = None
                    
            except Exception:
                # If GP fails, fall back to random or local
                break
                
            # Candidate points for acquisition
            # We can't optimize EI analytically easily. 
            # Strategy: Sample a set of candidate points (e.g., 100 random points in the box)
            # and pick the one with highest EI.
            # This is "Sample-based EI".
            
            n_candidates = 100
            # Generate candidates: mix of random and local perturbations around best
            cand_X = []
            
            # 1. Random points
            for _ in range(50):
                cand_X.append(np.random.uniform(-4.5, 4.5, dim))
                
            # 2. Local perturbations around current best
            for _ in range(50):
                pert = np.random.normal(0, 0.5, dim)
                cand_X.append(np.clip(best_x + pert, -5.0, 5.0))
                
            cand_X = np.array(cand_X)
            
            # Evaluate EI for candidates
            # K_star = compute_kernel(cand_X, X_data, length_scale)
            # mu = K_star @ alpha
            # K_ss = compute_kernel(cand_X, cand_X, length_scale)
            # var = np.diag(K_ss) - np.sum(K_star @ (K_star.T @ L_inv.T) ** 2, axis=1) ... complex
            # Simpler: var(x) = 1 - K(x, X) @ K^-1 @ K(X, x)
            # Since we have L, K^-1 = L^-T L^-1
            # var(x) = 1 - K_star @ (L_inv.T @ (L_inv @ K_star.T))
            
            K_star = compute_kernel(cand_X, X_data, length_scale)
            mu = K_star @ alpha
            
            if L_inv is not None:
                # var = 1 - K_star @ (L_inv.T @ (L_inv @ K_star.T))
                # Note: K_star is (n_cand, n_train)
                # L_inv is (n_train, n_train)
                # K_star @ L_inv.T is (n_cand, n_train)
                # Let V = L_inv @ K_star.T (n_train, n_cand)
                # var = 1 - np.sum(V * (K_star @ L_inv.T), axis=1)
                V = L_inv @ K_star.T
                var = 1.0 - np.sum(V * (K_star @ L_inv.T), axis=1)
            else:
                # Fallback
                K_inv = np.linalg.pinv(K)
                K_XX = compute_kernel(X_data, X_data, length_scale)
                var = 1.0 - np.sum(K_star @ (K_inv @ K_star.T), axis=1)
                
            var = np.maximum(var, 0.0)
            sigma = np.sqrt(var)
            
            ei = expected_improvement(mu, sigma, best_f, xi=0.01)
            
            # Find best candidate
            idx_best_cand = np.argmax(ei)
            x_next = cand_X[idx_best_cand]
            
            # Evaluate
            f_next = timed_call(x_next)
            
            # Update Data
            X_data = np.vstack([X_data, x_next])
            y_data = np.append(y_data, f_next)
            
            if f_next < best_f - 1e-8:
                best_f = f_next
                best_x = x_next.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # If stagnation is high, try a local optimizer (Nelder-Mead)
            # Nelder-Mead is fast for 5D.
            if stagnation_counter >= max_stagnation and n_calls < budget - 20:
                # Run Nelder-Mead from best_x
                def scipy_func(x):
                    if time.time() > deadline:
                        return float('inf')
                    return timed_call(x)
                    
                try:
                    res = scipy.optimize.minimize(scipy_func, best_x, method='Nelder-Mead', 
                                                 options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        X_data = np.vstack([X_data, best_x])
                        y_data = np.append(y_data, best_f)
                        stagnation_counter = 0
                except Exception:
                    pass
                
            # Adaptive Length Scale?
            # If we are improving, maybe keep LS. If not, increase LS to explore more?
            # For now, keep it fixed to avoid complexity.
            
        return best_x, best_f

    def __call__(self, func):
        # Re-implementing cleanly to avoid the "partial logic" issue from the thought process
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Halton Init
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def generate_halton_points(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                for j in range(dim):
                    base = primes[j]
                    point.append(halton_sequence(i, base))
                points.append(point)
            return np.array(points)

        n_init = 40
        init_points = generate_halton_points(n_init, dim)
        init_points = init_points * 9.0 - 4.5
        
        X_data = []
        y_data = []
        
        for x in init_points:
            if n_calls >= budget or time.time() > deadline:
                break
            val = timed_call(x)
            X_data.append(x)
            y_data.append(val)
            
        if not X_data:
            x0 = np.zeros(dim)
            val0 = timed_call(x0)
            X_data.append(x0)
            y_data.append(val0)
            
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 2. GP Surrogate
        def compute_kernel(X1, X2, length_scale=1.0):
            sq_dist = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-0.5 * sq_dist / (length_scale ** 2))
        
        length_scale = 1.0
        noise = 1e-8
        
        from scipy.stats import norm
        
        def expected_improvement(mu, sigma, y_best, xi=0.01):
            z = (mu - y_best - xi) / (sigma + 1e-8)
            return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        stagnation_counter = 0
        max_stagnation = 10
        
        while n_calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break
                
            try:
                K = compute_kernel(X_data, X_data, length_scale) + noise * np.eye(len(X_data))
                K += 1e-6 * np.eye(len(X_data))
                
                L = np.linalg.cholesky(K)
                v = np.linalg.solve(L, y_data)
                alpha = np.linalg.solve(L.T, v)
                L_inv = np.linalg.inv(L)
            except Exception:
                # Fallback: random search
                while n_calls < budget and time.time() < deadline:
                    x_rand = np.random.uniform(-4.5, 4.5, dim)
                    f_rand = timed_call(x_rand)
                    X_data = np.vstack([X_data, x_rand])
                    y_data = np.append(y_data, f_rand)
                break
                
            # Candidate selection
            n_candidates = 100
            cand_X = []
            for _ in range(50):
                cand_X.append(np.random.uniform(-4.5, 4.5, dim))
            for _ in range(50):
                pert = np.random.normal(0, 0.5, dim)
                cand_X.append(np.clip(best_x + pert, -5.0, 5.0))
                
            cand_X = np.array(cand_X)
            
            K_star = compute_kernel(cand_X, X_data, length_scale)
            mu = K_star @ alpha
            
            V = L_inv @ K_star.T
            var = 1.0 - np.sum(V * (K_star @ L_inv.T), axis=1)
            var = np.maximum(var, 0.0)
            sigma = np.sqrt(var)
            
            ei = expected_improvement(mu, sigma, best_f, xi=0.01)
            
            idx_best_cand = np.argmax(ei)
            x_next = cand_X[idx_best_cand]
            
            f_next = timed_call(x_next)
            
            X_data = np.vstack([X_data, x_next])
            y_data = np.append(y_data, f_next)
            
            if f_next < best_f - 1e-8:
                best_f = f_next
                best_x = x_next.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if stagnation_counter >= max_stagnation and n_calls < budget - 20:
                def scipy_func(x):
                    if time.time() > deadline:
                        return float('inf')
                    return timed_call(x)
                    
                try:
                    res = scipy.optimize.minimize(scipy_func, best_x, method='Nelder-Mead', 
                                                 options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        X_data = np.vstack([X_data, best_x])
                        y_data = np.append(y_data, best_f)
                        stagnation_counter = 0
                except Exception:
                    pass
            
        return best_x, best_f
