import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc, norm
from scipy.interpolate import RBFInterpolator

class Optimizer:
    """
    Optimizer using Gaussian Process surrogate with Expected Improvement.
    1. Initial exploration with LHS + Sobol + concentric rings for Schaffers structure.
    2. Fit RBF surrogate, use EI acquisition with proper uncertainty modeling.
    3. Adaptive local polish with multi-start Nelder-Mead and final high-precision pass.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
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

        # Phase 1: Initial Exploration - multi-strategy for Schaffers F7 structure
        n_explore = min(40, total_budget)
        
        points = []
        
        # LHS for broad coverage
        n_lhs = max(1, n_explore // 2)
        sampler = qmc.LatinHypercube(d=n, seed=42)
        samples = sampler.random(n_lhs)
        points.extend([(-5 + 10 * s) for s in samples])
        
        # Sobol for quasi-random uniform coverage
        n_sobol = max(1, n_explore // 3)
        sobol = qmc.Sobol(d=n, scramble=True, seed=123)
        sobol_samples = sobol.random(n_sobol)
        points.extend([(-5 + 10 * s) for s in sobol_samples])
        
        # Concentric ring points around origin - Schaffers F7 has valley structure
        n_ring = max(1, n_explore - n_lhs - n_sobol)
        if n_ring > 0:
            # Rings at different radii to capture the "bullseye" structure
            radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            for r in radii[:min(n_ring, len(radii))]:
                # Generate points on a sphere of radius r in n dimensions
                if r <= 5.0:
                    # Simple: scale a random unit vector
                    v = np.random.randn(n)
                    v = v / np.linalg.norm(v) * r
                    if np.all(np.abs(v) <= 5.0):
                        points.append(v)
        
        # Deduplicate and evaluate
        seen = set()
        for p in points:
            if calls >= total_budget or time.time() > deadline:
                break
            p = np.asarray(p, dtype=np.float64)
            p = clip(p)
            key = tuple(np.round(p, 4))
            if key in seen:
                continue
            seen.add(key)
            eval_func(p)

        # Phase 2: Bayesian Optimization Loop with RBF surrogate
        iteration = 0
        polish_interval = 20  # more frequent polish
        last_polish_f = np.inf
        
        while calls < total_budget and time.time() < deadline:
            iteration += 1
            
            # Check if we have enough data to fit surrogate
            if len(X_data) < 5:
                # Not enough data, add random points
                n_add = min(10, total_budget - calls)
                for _ in range(n_add):
                    if calls >= total_budget or time.time() > deadline:
                        break
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                continue
            
            # Fit RBF surrogate
            X_arr = np.array(X_data)
            f_arr = np.array(f_data)
            
            # Normalize for better conditioning
            f_mean = np.mean(f_arr)
            f_std = np.std(f_arr) if np.std(f_arr) > 1e-10 else 1.0
            f_norm = (f_arr - f_mean) / f_std
            
            try:
                # Use RBF with adaptive smoothing based on data spread
                spread = np.mean(np.linalg.norm(X_arr - X_arr.mean(axis=0), axis=1))
                smoothing = max(1e-5, 0.01 * f_std / max(spread, 1e-10))
                interpolator = RBFInterpolator(
                    X_arr, f_norm, 
                    kernel='gaussian',
                    smoothing=smoothing
                )
            except Exception:
                try:
                    interpolator = RBFInterpolator(
                        X_arr, f_norm, 
                        kernel='thin_plate_spline',
                        smoothing=1e-3
                    )
                except Exception:
                    p = np.random.uniform(-5, 5, n)
                    eval_func(p)
                    continue
            
            f_best_norm = min(f_norm)
            
            # Generate diverse candidate points for acquisition
            n_candidates = 200
            candidates = []
            
            # Random uniform
            n_rand = n_candidates // 3
            candidates.extend(np.random.uniform(-5, 5, (n_rand, n)))
            
            # Perturb top-3 points with varying scales
            n_good = min(3, len(X_arr))
            good_indices = np.argsort(f_norm)[:n_good]
            for idx in good_indices:
                n_perturb = max(2, (n_candidates // 3) // n_good)
                # Multiple perturbation scales for exploration/exploitation
                for scale in [0.1, 0.3, 0.5]:
                    perturbations = np.random.normal(0, scale, (n_perturb, n))
                    perturbed = X_arr[idx] + perturbations
                    perturbed = np.clip(perturbed, -5, 5)
                    candidates.append(perturbed)
            
            # Add points along principal directions from best point
            n_dir = max(2, n_candidates // 10)
            for _ in range(n_dir):
                direction = np.random.randn(n)
                direction /= np.linalg.norm(direction)
                for dist in [0.2, 0.5, 1.0]:
                    p = best_x + dist * direction
                    p = clip(p)
                    candidates.append(p)
            
            candidates = np.vstack(candidates)
            candidates = candidates[:n_candidates]
            
            # Evaluate surrogate on candidates
            try:
                f_pred_norm = interpolator(candidates)
            except Exception:
                f_pred_norm = np.random.normal(f_best_norm, 1.0, len(candidates))
            
            # Compute uncertainty based on distance to nearest data points
            # and local density
            dists = np.array([np.min(np.linalg.norm(X_arr - c, axis=1)) for c in candidates])
            
            # Uncertainty: increases with distance, decreases with local data density
            max_dist = np.max(dists) if np.max(dists) > 1e-10 else 1.0
            local_density = np.array([np.sum(dists <= 0.5) for _ in range(len(candidates))])
            uncertainty = 0.3 + 0.5 * (dists / max_dist) - 0.1 * local_density / max(1, np.max(local_density))
            uncertainty = np.clip(uncertainty, 0.05, 1.0)
            
            # EI for minimization
            diff = f_best_norm - f_pred_norm
            z = diff / (uncertainty + 1e-10)
            ei = diff * norm.cdf(z) + uncertainty * norm.pdf(z)
            
            # Add small randomization to break ties
            noise = np.random.normal(0, 0.003, len(candidates))
            scores = ei + noise
            
            # Select best candidate
            best_idx = np.argmax(scores)
            next_x = candidates[best_idx]
            
            # Evaluate the selected point
            eval_func(next_x)
            
            # Adaptive local polish: trigger when significant improvement or periodic
            remaining = total_budget - calls
            should_polish = False
            polish_budget = 0
            
            # Polish if we made progress since last polish
            if best_f < last_polish_f - 0.1 * f_std:
                should_polish = True
                last_polish_f = best_f
            
            # Or periodically
            if iteration % polish_interval == 0:
                should_polish = True
            
            # Reserve budget for final high-precision polish
            if remaining > 60 and calls > total_budget * 0.7:
                should_polish = False
            
            if should_polish and remaining > 20:
                budget_polish = min(30, remaining - 10)
                
                # Multi-start Nelder-Mead from top-2 points
                n_starts = min(2, len(X_data))
                top_indices = np.argsort(f_norm)[:n_starts]
                
                for start_idx in top_indices:
                    if calls >= total_budget - 10 or time.time() > deadline:
                        break
                    start_x = X_arr[start_idx].copy()
                    
                    def local_func(x):
                        if calls >= total_budget or time.time() > deadline:
                            raise ValueError("Budget Exceeded")
                        return eval_func(x)
                    
                    try:
                        res = minimize(local_func, start_x, method='Nelder-Mead',
                                       options={'maxiter': max(5, budget_polish // 3), 
                                                'xatol': 1e-6, 'fatol': 1e-6,
                                                'adaptive': True})
                    except Exception:
                        pass
                
                last_polish_f = best_f

        # Final high-precision polish pass
        remaining = total_budget - calls
        if remaining > 15 and time.time() < deadline - 5:
            final_budget = min(remaining - 5, 40)
            
            def final_func(x):
                if calls >= total_budget or time.time() > deadline:
                    raise ValueError("Budget Exceeded")
                return eval_func(x)
            
            try:
                res = minimize(final_func, best_x.copy(), method='Nelder-Mead',
                               options={'maxiter': max(10, final_budget // 2), 
                                        'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True})
            except Exception:
                pass
        
        return best_x, best_f
