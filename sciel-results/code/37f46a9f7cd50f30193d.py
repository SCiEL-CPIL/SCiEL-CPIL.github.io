"""Hybrid GP-Bayesian Optimization + BFGS Local Search for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import math


class Optimizer:
    """
    Hybrid optimizer combining Gaussian Process Bayesian Optimization (global)
    with BFGS local search (local) for BBOB f19 (Griewank-Rosenbrock).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _matern_52_kernel(self, A, B, length_scale):
        """Compute Matern 5/2 kernel between sets of points A and B."""
        # A: (m, d), B: (k, d)
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        sr = np.sqrt(5.0) * r / length_scale
        K = (1.0 + sr + sr * sr / 3.0) * np.exp(-sr)
        return K

    def _gp_predict(self, X, X_obs, y_obs, length_scale, noise):
        """
        Predict mean and variance at X using GP.
        X: (m, d) points to predict
        X_obs: (n, d) observed points
        y_obs: (n,) observed values
        Returns: mean (m,), var (m,)
        """
        n = X_obs.shape[0]
        if n == 0:
            return np.zeros(X.shape[0]), np.ones(X.shape[0])
        
        K = self._matern_52_kernel(X_obs, X_obs, length_scale)
        K[np.diag_indices_from(K)] += noise
        
        KX = self._matern_52_kernel(X_obs, X, length_scale)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            v = np.linalg.solve(L, KX)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            K[np.diag_indices_from(K)] += 1e-6
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
                v = np.linalg.solve(L, KX)
            except np.linalg.LinAlgError:
                # Fallback to zero variance if still failing
                return np.zeros(X.shape[0]), np.ones(X.shape[0])

        KXX = self._matern_52_kernel(X, X, length_scale)
        mean = KX.T @ alpha
        var = np.diag(KXX) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mean, var

    def _expected_improvement(self, mu, sigma, f_best, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        n = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # State for GP
        X_obs = []
        y_obs = []
        
        # Adaptive parameters
        length_scale = 2.0
        noise = 1e-6

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = func(x)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # --- Phase 1: Initial Exploration ---
        # Use a stratified LHS-like grid for initial coverage
        # For 5D, we can use a small grid or random points. 
        # Let's use 10 random points + origin for robust start.
        init_points = [np.zeros(n)]
        for _ in range(9):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        
        for p in init_points:
            if calls_made >= self.budget or time.time() > deadline - 20:
                break
            val = safe_call(p)
            if val is not None:
                X_obs.append(p)
                y_obs.append(val)

        # --- Phase 2: GP-Guided Search with Local BFGS Exploitation ---
        batch_size = 5
        bfgs_triggered = False
        
        while calls_made < self.budget and time.time() < deadline - 10:
            if len(X_obs) < 2:
                # Need at least 2 points for GP
                p = rng.uniform(-4.0, 4.0, n)
                val = safe_call(p)
                if val is not None:
                    X_obs.append(p)
                    y_obs.append(val)
                continue

            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            f_best_current = np.min(y_obs_arr)

            # Generate candidate points
            # Mix of random and local to best
            n_candidates = 150
            candidates = rng.uniform(-4.5, 4.5, (n_candidates, n))
            
            if best_x is not None:
                # Add candidates near the current best to help GP refine
                local_candidates = best_x + rng.normal(0, 0.5, (50, n))
                local_candidates = np.clip(local_candidates, -4.5, 4.5)
                candidates = np.vstack([candidates, local_candidates])

            # Predict GP
            mu, var = self._gp_predict(candidates, X_obs_arr, y_obs_arr, length_scale, noise)
            sigma = np.sqrt(var)
            
            # Compute EI
            ei = self._expected_improvement(mu, sigma, f_best_current, xi=0.005)
            
            # Select top batch_size candidates with diversity
            ei_sorted_idx = np.argsort(-ei)
            selected = []
            for idx in ei_sorted_idx:
                if len(selected) >= batch_size:
                    break
                cand = candidates[idx]
                # Diversity check
                if selected:
                    dists = np.array([np.linalg.norm(cand - s) for s in selected])
                    if np.min(dists) < 0.2:
                        continue
                selected.append(cand)
            
            # If not enough, fill with top EI
            if len(selected) < batch_size:
                for idx in ei_sorted_idx:
                    if len(selected) >= batch_size:
                        break
                    cand = candidates[idx]
                    if cand not in selected:
                        selected.append(cand)

            # Evaluate selected points
            for cand in selected:
                if calls_made >= self.budget or time.time() > deadline - 10:
                    break
                val = safe_call(cand)
                if val is not None:
                    X_obs.append(cand)
                    y_obs.append(val)

            # --- Local BFGS Exploitation ---
            # Trigger BFGS if we have made progress or are stuck
            # We trigger if the best point hasn't improved in last 10 calls 
            # or if we have enough budget left.
            if calls_made < self.budget - 20 and time.time() < deadline - 20:
                # Check if we should run BFGS
                # Simple heuristic: run BFGS from best_x every 20 calls if budget allows
                if (calls_made % 20 == 0) or (not bfgs_triggered and len(X_obs) > 10):
                    if best_x is not None:
                        bfgs_triggered = True
                        # Run BFGS from best_x
                        x0 = best_x.copy()
                        
                        def bfgs_func(x):
                            nonlocal calls_made, best_f, best_x
                            if calls_made >= self.budget or time.time() > deadline - 5:
                                return best_f
                            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                            val = func(x)
                            calls_made += 1
                            if val < best_f:
                                best_f = val
                                best_x = x.copy()
                            return val

                        try:
                            res = minimize(bfgs_func, x0, method='BFGS', 
                                           options={'maxiter': 30, 'gtol': 1e-6})
                            # Update best if BFGS found something better
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                                # Add this point to GP observations
                                X_obs.append(res.x)
                                y_obs.append(res.fun)
                        except Exception:
                            pass
                        bfgs_triggered = False # Reset to allow next trigger

            # Adapt length scale: shrink as we converge to exploit local structure
            if best_f < 1.0:
                length_scale = max(0.3, length_scale * 0.95)
            elif best_f < 10.0:
                length_scale = max(0.5, length_scale * 0.97)

        # --- Phase 3: Final Polish ---
        # If there's still budget, do a final (1+1)-ES or BFGS from best
        if calls_made < self.budget and time.time() < deadline - 2 and best_x is not None:
            polish_x = best_x.copy()
            polish_f = best_f
            polish_sigma = 0.05
            
            while calls_made < self.budget and time.time() < deadline - 1:
                step = polish_sigma * rng.standard_normal(n)
                candidate = np.clip(polish_x + step, -5, 5)
                candidate_f = func(candidate)
                calls_made += 1
                
                if candidate_f < polish_f:
                    polish_x = candidate
                    polish_f = candidate_f
                    polish_sigma *= 1.2
                    if polish_f < best_f:
                        best_f = polish_f
                        best_x = polish_x.copy()
                else:
                    polish_sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
