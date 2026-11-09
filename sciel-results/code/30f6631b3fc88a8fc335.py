"""
Strategy: Diagonal CMA-ES (Rank-1 + Rank-mu on diagonal covariance).
Rationale: The Discus function (f11) is highly anisotropic. Full CMA-ES is powerful
but computationally heavier and can be unstable. A diagonal covariance model
allows independent step sizes per dimension, rapidly adapting to the ellipsoidal
shape of the Discus landscape without the overhead of matrix decompositions.
This is structurally different: we model the search distribution as an axis-aligned
Gaussian with per-dimension sigmas, updated via natural gradient steps on the log-sigmas.
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        best_calls_made = 0

        def call_f(x):
            nonlocal n_calls, best_x, best_f, best_calls_made
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                best_calls_made = n_calls
            return f_val

        def diag_cma_es(x0, sigma0_vec, max_calls):
            nonlocal n_calls, best_x, best_f, best_calls_made
            
            # Population size
            lam = int(4 + 3 * np.log(dim))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Diagonal CMA-ES parameters
            # Adapt per-dimension sigmas
            c_sigma = 1 + 2 * max(0, np.sqrt((lam - 2) / (dim + 1)) - 1) / (dim + 4)
            c_1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (2 - c_sigma)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)) * (c_sigma - 1 / (dim + 2)))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            mean = x0.copy()
            sigmas = sigma0_vec.copy()  # Per-dimension step sizes
            
            start_calls = n_calls
            stagnation_counter = 0
            
            for _ in range(max_calls // lam + 10):
                if n_calls - start_calls >= max_calls:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Generate population: x = mean + sigmas * z
                ys = rng.standard_normal((lam, dim))
                xs = mean + np.outer(np.ones(lam), sigmas) * ys
                xs = np.clip(xs, -5.0, 5.0)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    f = call_f(xs[i])
                    if f is None:
                        return best_x, best_f
                    fs[i] = f
                
                # Sort by fitness
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                ys = ys[idx]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    best_calls_made = n_calls
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > 2 * lam:
                        break  # Stagnation detected
                
                # Recombine: weighted mean of top mu
                y_w = np.sum(w[:mu, np.newaxis] * ys[:mu], axis=0)
                x_w = mean + sigmas * y_w
                
                # Update per-dimension sigmas using natural gradient on log-sigmas
                # For diagonal C, the natural gradient of log-sigma_i is related to
                # the variance of the weighted y_i components.
                # Simple update: adapt sigmas based on the norm of y_w per dimension
                # and the population variance.
                
                # Compute per-dimension adaptation
                # The idea: if the weighted mean y_w has a large component in dimension i,
                # it suggests that dimension i is well-aligned, so we can shrink sigma_i.
                # If the population spread in dimension i is large, we might grow it.
                
                # Use a simple exponential update:
                # log(sigma_i) += c_sigma * (y_w_i / chi_N - 1)
                # This is a simplified version of the CMA-ES sigma update, applied per dimension.
                
                # Actually, for diagonal CMA-ES, the standard update for sigma is:
                # sigma *= exp(c_sigma / chi_N * (||p_sigma|| / chi_N - 1))
                # But p_sigma is a vector here. For diagonal, we can update each sigma independently
                # based on the corresponding component of the weighted y.
                
                # A robust approach: update each sigma_i based on the magnitude of y_w_i
                # and the expected magnitude under standard normal.
                # E[|z_i|] = sqrt(2/pi) ~ 0.798
                # If |y_w_i| > E[|z_i|], the search is moving in that direction, shrink sigma.
                # If |y_w_i| < E[|z_i|], the search is not moving, grow sigma.
                
                # Use a weighted update:
                # sigma_i *= exp(c_sigma * (y_w_i / chi_N - 1))
                # This can be unstable if y_w_i is negative. Use absolute value?
                # Better: use the squared magnitude to ensure positive update.
                # sigma_i *= exp(c_sigma * (y_w_i**2 / dim - 1))
                
                # Let's use a more principled approach:
                # The natural gradient for log-sigma_i in a diagonal CMA-ES is:
                # g_i = c_1 * (y_w_i^2 - 1) + c_mu * sum_{k=1}^{mu} w_k * (y_{k,i}^2 - 1)
                # Then: log(sigma_i) += g_i / (2 * mu_eff)  (approximate)
                
                # Simpler: just use the rank-mu update on log-sigmas
                rank_mu_vec = np.sum(w[:mu, np.newaxis] * ys[:mu]**2, axis=0)
                rank_one_vec = y_w**2
                
                # Update log-sigmas
                log_sigmas = np.log(sigmas)
                log_sigmas += (c_1 * rank_one_vec + c_mu * rank_mu_vec - 1) / (2 * mu_eff)
                
                # Clip to avoid extreme values
                log_sigmas = np.clip(log_sigmas, -10, 10)
                sigmas = np.exp(log_sigmas)
                
                # Update mean
                mean = x_w
                mean = np.clip(mean, -5.0, 5.0)
                
                # Check for convergence
                if np.min(sigmas) < 1e-12:
                    break

            return best_x, best_f

        # Multi-start Diagonal CMA-ES
        # Initial run from center
        x0 = np.zeros(dim)
        sigma0_vec = np.full(dim, 2.0)  # Initial step size, covering roughly half the box
        
        # Allocate budget
        # First run: 200 calls
        diag_cma_es(x0, sigma0_vec, 200)
        
        # Subsequent runs: smaller budgets, random starts
        while n_calls < budget and time.time() < deadline - 10:
            x_start = rng.uniform(-4, 4, dim)
            sigma_start_vec = rng.uniform(0.5, 2.0, dim)
            
            # Dynamic budget allocation
            remaining = budget - n_calls
            if remaining < 50:
                break
            
            # Use at most 100 calls per restart, but not more than remaining
            max_calls_for_run = min(100, remaining)
            
            diag_cma_es(x_start, sigma_start_vec, max_calls_for_run)
            
            # If we are not improving, break
            if n_calls - best_calls_made > 50 and best_f > 0: # Heuristic
                pass # Continue searching anyway as f11 is unimodal-ish
        
        return best_x, best_f
