import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """
    Optimizer using Bayesian Optimization with RBF surrogate.
    1. Initial exploration with Sobol sequence for uniform coverage.
    2. Iteratively fit an RBF surrogate and use Expected Improvement to select next points.
    3. Final high-precision polish with L-BFGS-B on the best point.
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
        n_explore = min(32, total_budget)
        
        if n_explore >= total_budget:
            points = np.random.uniform(-5, 5, (n_explore, n))
        else:
            # Use Sobol sequence for better uniform coverage in 5D
            sampler = qmc.Sobol(d=n, scramble=True, seed=42)
            samples = sampler.random(n_explore)
            points = -5 + 10 * samples
            # Add a few points near origin to help with centering
            points[-3:] = np.random.normal(0, 1, (3, n))
        
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            eval_func(p)

        # Phase 2: Bayesian Optimization Loop
        # Fit RBF surrogate and use EI acquisition
        polish_interval = 30  # every 30 BO iterations, do a local polish
        
        iteration = 0
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 4:
                # Not enough data, add random points
                n_add = min(8, total_budget - calls)
                for _ in range(n_add):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                continue
            
            # Fit RBF surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            try:
                # Use RBF interpolation as surrogate with thin plate spline
                interpolator = RBFInterpolator(
                    X_arr, f_arr, 
                    kernel='thin_plate_spline',
                    smoothing=1e-4
                )
            except Exception:
                try:
                    interpolator = RBFInterpolator(
                        X_arr, f_arr, 
                        kernel='gaussian',
                        smoothing=1e-3
                    )
                except Exception:
                    # Can't fit, do random search
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                    continue
            
            # Expected Improvement acquisition with uncertainty estimation
            f_best = min(f_arr)
            
            # Generate candidate points for acquisition
            n_candidates = 300
            # Mix of random and perturbed best points
            candidates = np.random.uniform(-5, 5, (n_candidates // 2, n))
            
            # Perturb best point and other good points with varying scales
            n_good = min(6, len(X_arr))
            good_indices = np.argsort(f_arr)[:n_good]
            for i, idx in enumerate(good_indices):
                # Use smaller perturbations for better points
                scale = 0.3 * (1 - i / max(1, n_good)) + 0.1
                n_perturb = max(2, n_candidates // 2 // n_good)
                perturbations = np.random.normal(0, scale, (n_perturb, n))
                perturbed = X_arr[idx] + perturbations
                perturbed = np.clip(perturbed, -5, 5)
                candidates = np.vstack([candidates, perturbed])
            
            # Truncate to max candidates
            candidates = candidates[:n_candidates]
            
            # Evaluate surrogate on candidates
            try:
                f_pred = interpolator(candidates)
            except Exception:
                f_pred = np.random.normal(f_best, 1.0, len(candidates))
            
            # Estimate uncertainty based on distance to nearest data points
            # Points far from data have higher uncertainty
            dists = np.min(np.linalg.norm(candidates[:, np.newaxis, :] - X_arr[np.newaxis, :, :], axis=2), axis=1)
            # Normalize distances
            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
            uncertainty = 0.5 * max_dist * (dists / max_dist) ** 0.5
            
            # Compute Expected Improvement with uncertainty
            # EI = (f_best - f_pred) * Phi((f_best - f_pred)/sigma) + sigma * phi((f_best - f_pred)/sigma)
            from scipy.stats import norm
            sigma = uncertainty + 1e-8
            z = (f_best - f_pred) / sigma
            ei = (f_best - f_pred) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best candidate by EI
            best_idx = np.argmax(ei)
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
                                                'xatol': 1e-6, 'fatol': 1e-6})
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
                               options={'maxiter': max(1, min(remaining // 2, 50)),
                                        'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                pass

        return best_x, best_f
