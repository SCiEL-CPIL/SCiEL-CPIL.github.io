import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc, norm

class Optimizer:
    """
    Optimizer using Bayesian Optimization with Gaussian Process surrogate.
    1. Initial exploration with Sobol sequence for uniform coverage.
    2. Iteratively fit a GP surrogate and use Expected Improvement to select next points.
    3. Adaptive restarts if progress stalls.
    4. Final high-precision polish with L-BFGS-B on the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        
        n = self.dim
        total_budget = self.budget
        
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Data storage for surrogate
        X_data = []
        f_data = []
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)

        # Evaluate function wrapper
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= total_budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (n,):
                x = x.reshape(n)
            x = clip(x)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls += 1
            if np.isfinite(fx):
                X_data.append(x.copy())
                f_data.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration with Sobol sequence
        n_explore = min(50, total_budget)
        
        if n_explore >= total_budget:
            points = np.random.uniform(-5, 5, (n_explore, n))
        else:
            # Use Sobol sequence for better uniform coverage in 5D
            sampler = qmc.Sobol(d=n, scramble=True, seed=42)
            samples = sampler.random(n_explore)
            points = -5 + 10 * samples
            # Add a few points near origin to help with centering
            points[-5:] = np.random.normal(0, 1.5, (5, n))
        
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            eval_func(p)

        # Phase 2: Bayesian Optimization Loop with GP surrogate
        polish_interval = 25  # every 25 BO iterations, do a local polish
        stall_threshold = 50  # if no improvement for this many iterations, restart
        last_improvement_iter = 0
        last_best_f = best_f
        
        iteration = 0
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 8:
                # Not enough data, add random points
                n_add = min(10, total_budget - calls)
                for _ in range(n_add):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                continue
            
            # Stall detection: if no improvement for stall_threshold iterations, restart
            if best_f < last_best_f - 1e-8:
                last_improvement_iter = iteration
                last_best_f = best_f
            elif iteration - last_improvement_iter > stall_threshold and iteration > 100:
                # Reset data and restart with new random points around best
                X_data.clear()
                f_data.clear()
                n_restart = min(15, total_budget - calls)
                for _ in range(n_restart):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = best_x + np.random.normal(0, 2.0, n)
                    p = np.clip(p, -5, 5)
                    eval_func(p)
                last_improvement_iter = iteration
                continue
            
            # Fit GP surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            # Standardize f for better GP conditioning
            f_mean = np.mean(f_arr)
            f_std = np.std(f_arr) + 1e-8
            f_std_arr = (f_arr - f_mean) / f_std
            
            try:
                # Simple GP with Matern kernel
                # Compute pairwise distances
                X_norm = (X_arr - np.mean(X_arr, axis=0)) / (np.std(X_arr, axis=0) + 1e-8)
                
                # Matern 5/2 kernel
                def matern_kernel(X1, X2, length_scale=1.0):
                    d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
                    d = np.sqrt(np.maximum(d2, 0)) * length_scale
                    return np.exp(-d) * (1 + d + d**2 / 3)
                
                K = matern_kernel(X_norm, X_norm, length_scale=0.5) + 1e-6 * np.eye(len(X_arr))
                
                # Solve for alpha
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_std_arr))
                except np.linalg.LinAlgError:
                    K += 1e-4 * np.eye(len(X_arr))
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_std_arr))
                
            except Exception:
                # Fallback: use simple inverse distance weighting
                p = np.random.uniform(-5, 5, n)
                eval_func(p)
                continue
            
            # Expected Improvement acquisition
            f_best = min(f_arr)
            f_best_std = (f_best - f_mean) / f_std
            
            # Generate candidate points for acquisition
            n_candidates = 400
            # Mix of random and perturbed best points
            candidates = np.random.uniform(-5, 5, (n_candidates // 2, n))
            
            # Perturb best point and other good points with varying scales
            n_good = min(8, len(X_arr))
            good_indices = np.argsort(f_arr)[:n_good]
            for i, idx in enumerate(good_indices):
                # Use smaller perturbations for better points
                scale = 0.5 * (1 - i / max(1, n_good)) + 0.1
                n_perturb = max(3, n_candidates // 2 // n_good)
                perturbations = np.random.normal(0, scale, (n_perturb, n))
                perturbed = X_arr[idx] + perturbations
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Normalize candidates
            X_cand_norm = (candidates - np.mean(X_arr, axis=0)) / (np.std(X_arr, axis=0) + 1e-8)
            
            # Compute GP predictions
            try:
                K_s = matern_kernel(X_cand_norm, X_norm, length_scale=0.5)
                mu_std = K_s @ alpha  # in standardized space
                v = np.linalg.solve(L, K_s.T)
                sigma2 = 1.0 - np.sum(v**2, axis=1)  # variance in standardized space
                sigma = np.sqrt(np.maximum(sigma2, 0))
                
                # Convert back to original scale
                mu = mu_std * f_std + f_mean
                sigma = sigma * f_std
                
                # Compute Expected Improvement
                # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
                z = (f_best - mu) / (sigma + 1e-8)
                ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                
            except Exception:
                # Fallback: use distance-based heuristic
                dists = np.min(np.linalg.norm(candidates[:, np.newaxis, :] - X_arr[np.newaxis, :, :], axis=2), axis=1)
                f_pred = f_best + 0.1 * dists
                sigma = 0.5 * dists + 0.1
                z = (f_best - f_pred) / (sigma + 1e-8)
                ei = (f_best - f_pred) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best candidate by EI
            best_idx = np.argmax(ei)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Periodic local polish
            if iteration % polish_interval == 0 and calls < total_budget - 20:
                # Do a quick Nelder-Mead polish from best_x
                remaining = total_budget - calls
                if remaining > 20:
                    budget_polish = min(20, remaining - 5)
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, best_x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(1, budget_polish // 3), 
                                                'xatol': 1e-5, 'fatol': 1e-5,
                                                'adaptive': True})
                    except Exception:
                        pass

        # Phase 3: Final high-precision polish with L-BFGS-B
        remaining = total_budget - calls
        if remaining > 10 and np.isfinite(best_f):
            def final_func(x):
                if calls >= total_budget or time.time() > deadline:
                    raise ValueError("Budget Exceeded")
                return eval_func(x)
            
            bounds = [(-5, 5)] * n
            try:
                res = minimize(final_func, best_x.copy(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': max(1, min(remaining // 2, 60)),
                                        'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxls': 20})
            except Exception:
                pass
            
            # If L-BFGS-B didn't improve much, try a second polish from a slightly perturbed point
            remaining2 = total_budget - calls
            if remaining2 > 5 and np.isfinite(best_f):
                def final_func2(x):
                    if calls >= total_budget or time.time() > deadline:
                        raise ValueError("Budget Exceeded")
                    return eval_func(x)
                
                try:
                    x_start = best_x + np.random.normal(0, 0.1, n)
                    x_start = np.clip(x_start, -5, 5)
                    res2 = minimize(final_func2, x_start, method='L-BFGS-B',
                                    bounds=bounds,
                                    options={'maxiter': max(1, min(remaining2, 20)),
                                             'ftol': 1e-12, 'gtol': 1e-10})
                except Exception:
                    pass

        return best_x, best_f
