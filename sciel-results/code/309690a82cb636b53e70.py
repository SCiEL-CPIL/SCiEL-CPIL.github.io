"""Optimizer for BBOB f7 (Step ellipsoidal) using CMA-ES with adaptive sigma."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for Step ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill
        d = self.dim
        max_calls = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        # Population size
        pop_size = int(4 + 3 * np.log(d))  # standard CMA-ES population size
        
        # Initial parameters
        x_mean = np.zeros(d)  # Start at center
        sigma = 1.0  # Reduced from 2 to 1 for better step boundary resolution
        
        # Covariance matrix
        C = np.eye(d)
        
        # Step size for mean update
        alpha_mu = 1.0
        cmu = 2 / ((d + 1.7)**2 + pop_size)
        ccs = (4 + d / 4) / (d + 4 + 2 * d / (d + 1.7))
        c1 = (2 + 2 / (d + 1.7)) / ((d + 2)**2 + pop_size)
        cmu = min(0.4, (2 + 2 / (d + 1.7)) / ((d + 2)**2 + pop_size))
        
        # Path accumulators
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Normalization constants
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Number of function evaluations so far
        n_evals = 0
        
        # Initial population: sample from N(0, sigma^2 * I)
        def sample_population():
            nonlocal calls_made, best_x, best_f
            samples = []
            for i in range(pop_size):
                if not elapsed_ok():
                    break
                z = rng.standard_normal(d)
                x = x_mean + sigma * z
                fx = evaluate(x)
                samples.append((fx, x))
            return samples

        # Main CMA-ES loop
        while elapsed_ok():
            # Sample population
            samples = sample_population()
            if len(samples) < pop_size:
                break
            
            # Sort by fitness
            samples.sort(key=lambda t: t[0])
            
            # Update best
            if samples[0][0] < best_f:
                best_f = samples[0][0]
                best_x = samples[0][1].copy()
            
            # Select top lambda/2 for mean update
            mu = pop_size // 2
            weights = np.ones(mu)
            # Use equal weights for simplicity, could use log-spaced
            w_sum = np.sum(weights)
            weights = weights / w_sum
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Update mean
            x_new = np.zeros(d)
            for i in range(mu):
                x_new += weights[i] * samples[i][1]
            
            # Save old mean
            x_old = x_mean.copy()
            
            # Update sigma (path adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma) * (C @ (x_new - x_old))
            
            # Check for delta rule
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) > (1.4 + 1/d) * chi_d:
                sigma *= 1.0  # No change for simplicity
            
            # Update covariance matrix
            v = x_new - x_old
            C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T) + cmu * np.sum([weights[i] * np.outer(samples[i][1] - x_old, samples[i][1] - x_old) for i in range(mu)], axis=0)
            
            # Update path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (v / sigma)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            
            # Eigenvalue decomposition for sampling
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
            except:
                B = np.eye(d)
            
            # Update mean
            x_mean = x_new
            
            # Adapt sigma
            sigma *= np.exp((c_sigma / d) * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # If sigma is too small, do a random restart
            if sigma < 1e-4:
                # Random restart around best
                if elapsed_ok():
                    x_restart = best_x + rng.standard_normal(d) * 0.5
                    fx = evaluate(x_restart)
                    if fx < best_f:
                        best_f = fx
                        best_x = x_restart.copy()
                        x_mean = best_x
                        sigma = 0.5
                else:
                    break

        # Final local search around best point
        if elapsed_ok():
            # Try small perturbations around best
            for trial in range(20):
                if not elapsed_ok():
                    break
                sigma_local = 0.1 * (0.9 ** trial)
                y = best_x + rng.standard_normal(d) * sigma_local
                fy = evaluate(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
