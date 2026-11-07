import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """
    Optimizer using Gaussian Process surrogate with Expected Improvement.
    1. Initial exploration with Sobol sequence for uniform coverage.
    2. Fit GP surrogate, use EI acquisition for model-guided search.
    3. Final local polish with Nelder-Mead and L-BFGS-B.
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
        # Use Sobol for better uniform coverage than LHS in 5D
        n_explore = min(40, total_budget)
        
        if n_explore >= total_budget:
            points = np.random.uniform(-5, 5, (n_explore, n))
        else:
            # Sobol sequence for low-discrepancy sampling
            sampler = qmc.Sobol(d=n, scramble=True, seed=42)
            samples = sampler.random(n_explore)
            points = -5 + 10 * samples
        
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            eval_func(p)

        # Phase 2: GP-based Bayesian Optimization Loop
        polish_interval = 50  # every 50 BO iterations, do a local polish
        
        iteration = 0
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 5:
                # Not enough data, add more Sobol points
                n_add = min(10, total_budget - calls)
                if n_add > 0:
                    sampler = qmc.Sobol(d=n, scramble=True, seed=100+iteration)
                    samples = sampler.random(n_add)
                    points = -5 + 10 * samples
                    for p in points:
                        if calls >= total_budget or time.time() > deadline:
                            break
                        eval_func(p)
                continue
            
            # Fit GP surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            # Normalize f for numerical stability
            f_mean = np.mean(f_arr)
            f_std = np.std(f_arr) + 1e-10
            f_norm = (f_arr - f_mean) / f_std
            
            try:
                # Use RBF interpolation as surrogate with Gaussian kernel
                interpolator = RBFInterpolator(
                    X_arr, f_norm, 
                    kernel='gaussian',
                    smoothing=1e-2
                )
            except Exception:
                # Fallback: do random search
                p = np.random.uniform(-5, 5, n)
                eval_func(p)
                continue
            
            # Generate candidate points for acquisition
            n_candidates = 300
            # Mix of random and perturbed best points
            candidates = np.random.uniform(-5, 5, (n_candidates // 3, n))
            
            # Perturb best point and other good points
            n_good = min(8, len(X_arr))
            good_indices = np.argsort(f_arr)[:n_good]
            for idx in good_indices:
                n_perturb = max(2, n_candidates // 3 // n_good)
                # Use decreasing perturbation scale based on how good the point is
                rank = np.where(good_indices == idx)[0][0]
                scale = 1.0 / (1.0 + rank * 0.5)
                perturbations = np.random.normal(0, 0.8 * scale, (n_perturb, n))
                perturbed = X_arr[idx] + perturbations
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Add points near the global best with smaller perturbations
            n_fine = 50
            fine_perturbations = np.random.normal(0, 0.3, (n_fine, n))
            fine_perturbed = best_x + fine_perturbations
            fine_perturbed = np.clip(fine_perturbed, -5, 5)
            candidates = np.vstack([candidates, fine_perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Evaluate surrogate on candidates
            try:
                f_pred_norm = interpolator(candidates)
                f_pred = f_pred_norm * f_std + f_mean
            except Exception:
                f_pred = np.random.normal(f_mean, f_std, len(candidates))
            
            # Compute Expected Improvement with approximate uncertainty
            # Use distance to nearest data point as a proxy for uncertainty
            f_best = min(f_arr)
            
            # Compute distances to nearest data points (approximate uncertainty)
            from scipy.spatial import cKDTree
            try:
                tree = cKDTree(X_arr)
                dists, _ = tree.query(candidates, k=1)
                # Scale distance to approximate uncertainty
                max_dist = np.max(np.linalg.norm(X_arr - best_x, axis=1)) + 1e-10
                uncertainty = 0.5 * f_std * (dists / max_dist)
            except Exception:
                uncertainty = np.full(len(candidates), 0.1 * f_std)
            
            # EI = (f_best - f_pred) * Phi((f_best - f_pred)/sigma) + sigma * phi((f_best - f_pred)/sigma)
            sigma = uncertainty + 1e-8
            z = (f_best - f_pred) / sigma
            ei = (f_best - f_pred) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Add small randomization to avoid getting stuck
            noise = np.random.normal(0, 0.005 * max(1, abs(f_best)), len(candidates))
            scores = ei + noise
            
            # Select best candidate
            best_idx = np.argmax(scores)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Periodic local polish
            if iteration % polish_interval == 0 and calls < total_budget - 30:
                # Do a quick Nelder-Mead polish from best_x
                remaining = total_budget - calls
                if remaining > 30:
                    budget_polish = min(25, remaining - 10)
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, best_x.copy(), method='Nelder-Mead',
                                       options={'maxiter': max(1, budget_polish // 5), 
                                                'xatol': 1e-10, 'fatol': 1e-10})
                        if res.success and res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 3: Final local polish with L-BFGS-B if budget allows
        if calls < total_budget - 10 and time.time() < deadline - 5:
            remaining = total_budget - calls
            if remaining > 10:
                budget_final = min(15, remaining - 2)
                
                def final_func(x):
                    if calls >= total_budget or time.time() > deadline:
                        raise ValueError("Budget Exceeded")
                    return eval_func(x)
                
                try:
                    res = minimize(final_func, best_x.copy(), method='L-BFGS-B',
                                   bounds=[(-5, 5)] * n,
                                   options={'maxiter': max(1, budget_final // 3),
                                            'ftol': 1e-12, 'gtol': 1e-12})
                    if res.success and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
