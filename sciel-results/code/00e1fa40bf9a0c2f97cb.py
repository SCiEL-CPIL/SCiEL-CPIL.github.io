import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.utils import check_array
import warnings

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        calls = 0
        best_x = None
        best_f = float('inf')
        dim = self.dim
        
        # Storage for surrogate model
        X_samples = []
        Y_samples = []
        
        # Helper to evaluate and record
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget or time.time() > deadline:
                return None
            x_eval = np.clip(x, -5, 5)
            fx = func(x_eval)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_eval.copy()
                
            # Record for surrogate
            X_samples.append(x_eval)
            Y_samples.append(fx)
            
            return fx
        
        # Phase 1: Initial Sampling (Latin Hypercube + Strategic Points)
        # We want a diverse set of initial points to build a decent GP model.
        # Budget: 500. Let's use 50 for initial sampling.
        initial_budget = 50
        
        # Strategic points: Origin, Nominal Min (1,1,1,1,1), and a few randoms
        strategic_points = [
            np.zeros(dim),
            np.ones(dim),
            -np.ones(dim),
            np.array([5.0]*dim),
            np.array([-5.0]*dim)
        ]
        
        # Generate Latin Hypercube points
        rng = np.random.default_rng(42)
        n_lhc = initial_budget - len(strategic_points)
        
        # Simple LHC generation
        lhc_points = []
        for i in range(n_lhc):
            p = np.zeros(dim)
            for d in range(dim):
                p[d] = -5 + 10 * (i + rng.random()) / n_lhc
            lhc_points.append(p)
        
        all_initial = strategic_points + lhc_points
        
        # Shuffle to mix strategic and random early? No, strategic first is good for anchoring.
        # Actually, let's just evaluate them all.
        for p in all_initial:
            if time.time() > deadline or calls >= self.budget:
                break
            try:
                evaluate(p)
            except Exception:
                pass
        
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1
            X_samples.append(best_x)
            Y_samples.append(best_f)

        # Phase 2: Surrogate-Assisted Optimization
        # Use the remaining budget to iteratively pick points via EI.
        
        # GP Setup
        # Matern kernel with nu=2.5 is good for smooth but non-analytic functions like Rosenbrock.
        # Length scales: Initialize based on domain size.
        # We use a fixed kernel structure for speed, but let sklearn optimize length scales if possible?
        # Optimizing kernel params is expensive. Let's use a fixed kernel with reasonable length scales.
        # Domain is [-5, 5]^5. Length scale ~ 2.0 seems reasonable for initial exploration.
        
        kernel = Matern(length_scale=np.ones(dim) * 2.0, nu=2.5)
        
        # We will retrain the GP every few iterations or use incremental? 
        # sklearn GP doesn't support incremental. Retraining on 50-100 points in 5D is fast (<10ms).
        # So we can retrain frequently.
        
        # Acquisition: Expected Improvement
        # We need to handle the "best_f" in EI.
        
        def expected_improvement(X, gpr, y_best):
            """
            Calculate Expected Improvement for a set of points X.
            """
            try:
                y_pred, y_std = gpr.predict(X, return_std=True)
                y_std = np.maximum(y_std, 1e-8)
                
                # EI = (y_best - y_pred) * Phi((y_best - y_pred) / y_std) + y_std * phi((y_best - y_pred) / y_std)
                # where Phi is CDF of standard normal, phi is PDF.
                from scipy.stats import norm
                
                Z = (y_best - y_pred) / y_std
                ei = (y_best - y_pred) * norm.cdf(Z) + y_std * norm.pdf(Z)
                return ei
            except Exception:
                return np.zeros(len(X))

        # Main Loop
        # We have ~450 calls left.
        # Strategy: 
        # 1. Train GP on current data.
        # 2. Generate a set of candidate points (e.g., 100 random points in the box).
        # 3. Calculate EI for these candidates.
        # 4. Pick the top K (e.g., 5) candidates.
        # 5. Evaluate them.
        # 6. Repeat until budget is exhausted.
        
        # To speed up, we can batch evaluate.
        
        # How often to retrain? Every iteration (every K evals) is fine.
        # How many candidates to consider? 50-100 is fast for GP prediction.
        
        num_candidates = 100
        num_selects = 5
        
        while calls < self.budget and time.time() < deadline:
            # Check if we have enough data to train GP (min 2 points)
            if len(X_samples) < 2:
                # Just do random search if data is too sparse
                x_new = rng.uniform(-5, 5, dim)
                evaluate(x_new)
                continue
            
            # Train GP
            X_train = np.array(X_samples)
            Y_train = np.array(Y_samples)
            
            # Standardize Y for better GP conditioning? 
            # Rosenbrock values can vary. Let's try standardizing.
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_train_scaled = (Y_train - y_mean) / y_std
            
            try:
                gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0)
                gpr.fit(X_train, Y_train_scaled)
            except Exception:
                # If GP fails, fall back to local search or random
                x_new = rng.uniform(-5, 5, dim)
                evaluate(x_new)
                continue
            
            # Generate candidates
            # Mix of random points and points near current best
            # 80% random in box, 20% local around best
            n_rand = int(num_candidates * 0.8)
            n_local = num_candidates - n_rand
            
            candidates = []
            
            # Random
            rand_pts = rng.uniform(-5, 5, (n_rand, dim))
            candidates.extend(rand_pts.tolist())
            
            # Local around best
            if best_x is not None:
                local_sigma = 1.0 # Initial local exploration radius
                local_pts = best_x + local_sigma * rng.standard_normal((n_local, dim))
                local_pts = np.clip(local_pts, -5, 5)
                candidates.extend(local_pts.tolist())
            else:
                candidates.extend(rng.uniform(-5, 5, (n_local, dim)).tolist())
            
            candidates = np.array(candidates)
            
            # Calculate EI
            # Note: gpr.predict expects scaled Y? No, gpr was fit on scaled Y.
            # So y_pred will be in scaled space.
            # We need y_best in scaled space for EI calculation.
            y_best_scaled = (best_f - y_mean) / y_std
            
            ei = expected_improvement(candidates, gpr, y_best_scaled)
            
            # Select top K
            top_indices = np.argsort(ei)[-num_selects:]
            
            # Evaluate selected points
            for idx in top_indices:
                if calls >= self.budget or time.time() > deadline:
                    break
                x_new = candidates[idx]
                evaluate(x_new)
                
                # If we found a significantly better point, we might want to shrink local_sigma?
                # Or just let the GP handle it.
                
        # Phase 3: Final Local Refinement
        # Use remaining budget for a quick Powell run from best_x
        if calls < self.budget and best_x is not None:
            remaining = self.budget - calls
            if remaining > 10:
                def wrapper(x_eval):
                    nonlocal calls, best_x, best_f
                    if calls >= self.budget or time.time() > deadline:
                        raise Exception("Stop")
                    x_eval = np.clip(x_eval, -5, 5)
                    fx = func(x_eval)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_eval.copy()
                    return fx
                
                try:
                    minimize(wrapper, best_x, method='Powell', options={'maxfev': remaining, 'maxiter': 50})
                except Exception:
                    pass
        
        return best_x, best_f
