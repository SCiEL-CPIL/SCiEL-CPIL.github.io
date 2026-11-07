import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Global Search with Differential Evolution
        global_budget = int(0.5 * self.budget)
        
        popsize = 10
        max_iter_de = max(1, int(global_budget / popsize) - 1)
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds=[(-5, 5)] * self.dim, 
                maxiter=max_iter_de,
                popsize=popsize,
                polish=False,
                tol=1e-8,
                seed=0,
                init='latinhypercube'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x
        except (TimeoutError, Exception):
            pass

        # Phase 2: RBF Surrogate with Expected Improvement
        remaining_budget = self.budget - calls
        if remaining_budget > 10 and best_x is not None:
            # Collect points for RBF: use DE population + best
            # We'll sample points around the best and use EI to guide search
            
            # Initialize with points we've seen (we don't have full history, so sample)
            rng = np.random.default_rng(42)
            
            # Sample initial points for RBF training
            X_rbf = [best_x.copy()]
            F_rbf = [best_f]
            
            # Add some random points to build initial model
            for _ in range(min(20, remaining_budget // 5)):
                if not time_ok() or calls >= self.budget:
                    break
                x_sample = rng.uniform(-5, 5, size=dim)
                f_sample = wrapped_func(x_sample)
                X_rbf.append(x_sample)
                F_rbf.append(f_sample)
            
            X_rbf = np.array(X_rbf)
            F_rbf = np.array(F_rbf)
            
            def rbf_predict(X, X_train, F_train, length_scale=1.5):
                """RBF with Gaussian kernel, return mean and std"""
                X = np.atleast_2d(X)
                # Compute pairwise distances
                diff = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                K = np.exp(-sq_dist / (2 * length_scale ** 2))
                
                # Simple weighted average for prediction
                weights = K / (K.sum(axis=1, keepdims=True) + 1e-10)
                pred = weights @ F_train
                # Uncertainty: inverse of weight concentration
                max_w = weights.max(axis=1)
                uncertainty = (1 - max_w) * (F_train.max() - F_train.min() + 1e-8)
                return pred, uncertainty
            
            def expected_improvement(X, X_train, F_train, best_val, kappa=0.01):
                """EI acquisition function"""
                pred, unc = rbf_predict(X, X_train, F_train)
                improvement = best_val - pred
                z = improvement / (unc + 1e-10)
                from scipy.stats import norm
                EI = improvement * norm.cdf(z) + unc * norm.pdf(z)
                EI = np.maximum(EI, 0)
                return EI
            
            # Iterative EI-based search
            x_current = best_x.copy()
            f_current = best_f
            
            for _ in range(remaining_budget):
                if not time_ok() or calls >= self.budget:
                    break
                
                # Sample candidate points
                n_candidates = 50
                candidates = []
                
                # Mix: local around current best, global random
                for i in range(n_candidates):
                    if i < n_candidates // 2:
                        # Local: around current best
                        x_cand = x_current + rng.normal(0, 0.5, size=dim)
                    elif i < 3 * n_candidates // 4:
                        # Medium: around best with larger radius
                        x_cand = best_x + rng.normal(0, 1.5, size=dim)
                    else:
                        # Global: random
                        x_cand = rng.uniform(-5, 5, size=dim)
                    x_cand = np.clip(x_cand, -5, 5)
                    candidates.append(x_cand)
                
                candidates = np.array(candidates)
                
                # Evaluate EI for each candidate
                ei_values = expected_improvement(candidates, X_rbf, F_rbf, f_current)
                
                # Select candidate with highest EI
                idx = np.argmax(ei_values)
                x_next = candidates[idx]
                
                # Evaluate
                f_next = wrapped_func(x_next)
                
                # Update RBF training data
                X_rbf = np.vstack([X_rbf, x_next])
                F_rbf = np.append(F_rbf, f_next)
                
                # Update best
                if f_next < f_current:
                    f_current = f_next
                    x_current = x_next.copy()
                
                # Periodically retrain or adjust length scale
                # Keep model size manageable
                if len(F_rbf) > 100:
                    # Keep best 50 points by function value
                    idx_sorted = np.argsort(F_rbf)[:50]
                    X_rbf = X_rbf[idx_sorted]
                    F_rbf = F_rbf[idx_sorted]

            # Update global best
            if f_current < best_f:
                best_f = f_current
                best_x = x_current

        # Phase 3: Final local polish with Nelder-Mead
        remaining = self.budget - calls
        if remaining > 5 and best_x is not None:
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 100), 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
