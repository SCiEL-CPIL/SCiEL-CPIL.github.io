import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(dim)

        # Helper to evaluate and track
        def evaluate(x):
            if self.calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            self.calls += 1
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Nelder-Mead implementation
        def run_nelder_mead(x_start, max_calls):
            local_best_x = x_start.copy()
            local_best_f = evaluate(x_start)
            local_calls = 1
            
            if local_calls >= max_calls or time.time() > deadline:
                return local_best_x, local_best_f, local_calls

            # Initial simplex
            simplex = [x_start.copy()]
            for i in range(dim):
                x_i = x_start.copy()
                x_i[i] += 0.5 if x_start[i] < 0 else -0.5
                simplex.append(x_i)
            
            simplex_f = [local_best_f]
            for i in range(1, dim + 1):
                simplex_f.append(evaluate(simplex[i]))
                local_calls += 1
                if simplex_f[i] < local_best_f:
                    local_best_f = simplex_f[i]
                    local_best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-8

            while local_calls < max_calls and time.time() < deadline:
                # Sort simplex
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()

                # Check convergence
                if local_calls >= max_calls - 1:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                local_calls += 1
                
                if f_r < local_best_f:
                    local_best_f = f_r
                    local_best_x = x_r.copy()

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    local_calls += 1
                    if f_e < local_best_f:
                        local_best_f = f_e
                        local_best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    local_calls += 1
                    if f_c < local_best_f:
                        local_best_f = f_c
                        local_best_x = x_c.copy()
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
                            local_calls += 1
                            if simplex_f[i] < local_best_f:
                                local_best_f = simplex_f[i]
                                local_best_x = simplex[i].copy()
                        
                        # Check if shrinkage was effective
                        if np.max(np.abs(simplex[1] - simplex[0])) < min_step:
                            break

            return local_best_x, local_best_f, local_calls

        # --- GP-Guided Multi-Start Strategy ---
        
        # Phase 1: Use Gaussian Process surrogate to guide exploration
        # The Discus function has a narrow valley structure, so GP can help map it efficiently
        
        from scipy.stats import norm
        
        # Collect initial samples
        X_samples = []
        f_samples = []
        
        # Start with diverse initial points
        initial_starts = [
            np.zeros(dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim),
            np.random.uniform(-4, 4, dim)
        ]
        
        for x0 in initial_starts:
            if self.calls >= budget - 100:
                break
            if time.time() > deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            f0 = evaluate(x0)
            X_samples.append(x0)
            f_samples.append(f0)
        
        # GP-based exploration phase
        gp_iterations = 0
        max_gp_iterations = 30
        best_x_gp = self.best_x.copy()
        best_f_gp = self.best_f
        
        while (self.calls < budget - 100 and 
               time.time() < deadline and 
               gp_iterations < max_gp_iterations):
            
            if len(X_samples) < 3:
                # Need at least 3 points for GP
                x_new = np.random.uniform(-4, 4, dim)
                f_new = evaluate(x_new)
                X_samples.append(x_new)
                f_samples.append(f_new)
                gp_iterations += 1
                continue
            
            X_arr = np.array(X_samples)
            f_arr = np.array(f_samples)
            
            # Fit GP
            # Use simple RBF kernel with length scale
            L = 1.0  # length scale
            sigma_f = np.std(f_arr) + 1e-6
            
            # Compute kernel matrix
            def rbf_kernel(X1, X2, L=L, sigma_f=sigma_f):
                # Compute squared distances
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                K = sigma_f**2 * np.exp(-0.5 * sq_dists / L**2)
                return K
            
            K = rbf_kernel(X_arr, X_arr) + 1e-6 * np.eye(len(X_arr))
            
            # Predictive mean and variance at candidate points
            # Use EI acquisition function
            # Generate candidate points on a grid
            n_candidates = 200
            candidates = np.random.uniform(-4.5, 4.5, (n_candidates, dim))
            
            K_x = rbf_kernel(X_arr, candidates)
            K_xx = rbf_kernel(candidates, candidates)
            
            # GP predictions
            try:
                K_inv = np.linalg.inv(K)
                f_mean = K_x.T @ K_inv @ f_arr
                f_var = np.diag(K_xx) - np.sum(K_x.T @ K_inv @ K_x, axis=0)
                f_var = np.maximum(f_var, 1e-10)
                f_std = np.sqrt(f_var)
            except np.linalg.LinAlgError:
                # Fallback to random point
                x_new = np.random.uniform(-4, 4, dim)
                f_new = evaluate(x_new)
                X_samples.append(x_new)
                f_samples.append(f_new)
                gp_iterations += 1
                continue
            
            # Expected Improvement
            current_best = np.min(f_arr)
            Z = (current_best - f_mean) / (f_std + 1e-10)
            EI = (current_best - f_mean) * norm.cdf(Z) + f_std * norm.pdf(Z)
            
            # Select point with max EI
            idx_best = np.argmax(EI)
            x_new = candidates[idx_best]
            f_new = evaluate(x_new)
            
            X_samples.append(x_new)
            f_samples.append(f_new)
            
            if f_new < best_f_gp:
                best_f_gp = f_new
                best_x_gp = x_new.copy()
            
            gp_iterations += 1
        
        # Phase 2: Exploit with Nelder-Mead from GP best
        if self.calls < budget - 10 and time.time() < deadline:
            remaining_budget = budget - self.calls
            
            # Run Nelder-Mead from the best point found by GP
            x_final, f_final, _ = run_nelder_mead(best_x_gp, remaining_budget)
            
            # The global best_x and best_f are already updated by 'evaluate' inside run_nelder_mead
            # So self.best_x and self.best_f are correct.

        return self.best_x, self.best_f
