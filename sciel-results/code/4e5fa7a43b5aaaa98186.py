"""CMA-ES with adaptive population size for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: try a few starting points, pick best
        x0 = np.zeros(n)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0

        # Try a few random starts to find a better initial point
        for i in range(5):
            xi = rng.uniform(-4, 4, n)
            fi = func(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        x = best_x.copy()
        f_best = best_f
        calls_used = 6  # 1 initial + 5 random

        # CMA-ES parameters
        # Adaptive population: start large, shrink as we converge
        # lambda_t = max(lambda_min, lambda_0 * (1 - t/T)^alpha)
        lambda_0 = max(4, n + 3)  # initial population size
        lambda_min = 2
        c_sigma = 1.0 / (n + 4)  # step size adaptation rate
        c1 = (2 / ((n + 1.3)**2 + 10))  # rank-one update
        cmu = min(0.5 - c1, 2 / ((n + 2)**2 - 4))  # rank-mu update
        c1 = (2 / ((n + 1.3)**2 + 10))
        cmu = min(0.5 - c1, 2 / ((n + 2)**2 - 4))
        cmu = max(cmu, 0.0)

        mu = max(1, lambda_0 // 2)  # number of recombination weights
        # Weights: linear weights, normalized
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)

        # Initial step size
        sigma = 2.0
        # Initial mean
        mean = x.copy()
        # Covariance matrix
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # Chi-square correction
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        # Adaptive population schedule
        # Total iterations: we'll do many iterations with shrinking lambda
        # Each iteration uses lambda_t evaluations
        # We want to use all budget

        # Strategy: 
        # Phase 1: Exploration with larger population
        # Phase 2: Exploitation with smaller population
        # Use adaptive lambda based on progress

        # We'll track how much budget we've used and adapt
        # lambda_t decreases as we approach the budget

        # Let's plan: total budget = budget
        # We'll do iterations until budget exhausted
        # lambda starts at lambda_0 and shrinks

        # Compute how many iterations we can afford
        # If lambda stays constant at lambda_0, we get budget // lambda_0 iterations
        # But lambda shrinks, so we get more iterations later

        # Simpler approach: dynamically choose lambda each iteration
        # based on remaining budget and convergence

        it = 0
        while calls_used < budget:
            it += 1
            # Determine population size for this iteration
            # Start with lambda_0, shrink as we converge
            # Use a schedule: lambda = max(lambda_min, int(lambda_0 * (1 - it/total_iters)^0.5))
            # But we don't know total_iters a priori
            # Instead: shrink based on progress (improvement in best_f)
            
            # Alternative: fixed schedule based on iteration count
            # Estimate total iterations assuming average lambda
            # Let's use: lambda_t = max(lambda_min, lambda_0 * exp(-it / tau))
            tau = max(10, budget // (2 * lambda_0))  # decay timescale
            lambda_t = max(lambda_min, int(lambda_0 * np.exp(-it / tau)))
            
            # Don't exceed remaining budget
            lambda_t = min(lambda_t, budget - calls_used)
            if lambda_t < 2:
                break
            
            mu_t = max(1, lambda_t // 2)
            # Recompute weights for current lambda_t
            w_t = np.log(mu_t + 0.5) - np.log(np.arange(1, mu_t + 1))
            w_t = w_t / w_t.sum()
            mueff_t = 1.0 / np.sum(w_t**2)
            
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_t):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(n)
                # Apply C_sqrt_inv
                y = mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_used += 1
                
                if fy < f_best:
                    f_best = fy
                    best_x = y.copy()
            
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Mean update
            mean_old = mean.copy()
            mean = np.sum(w_t[:, np.newaxis] * ys_sorted[:mu_t], axis=0)
            
            # Evolution path updates
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff_t) * (C_sqrt_inv @ (mean - mean_old) / sigma)
            
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(it+1)))) < (1.4 + 2/(n+1)) * chi_n
            
            c_c = (4 + mueff_t/n) / (n + 4 + 2*mueff_t/n)
            c_c = max(c_c, 0.1)
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff_t) * (mean - mean_old) / sigma
            
            # Covariance matrix update
            delta_hsig = (1.0 if hsig else 0.0) - (1 - c_c)
            diff = mean - mean_old
            C = (1 + c_sigma * c_sigma * mueff_t * (hsig - 1) - c1 - cmu) * C \
                + c1 * np.outer(p_c, p_c) + cmu * np.sum(w_t[:mu_t] * np.outer(ys_sorted[:mu_t] - mean_old, ys_sorted[:mu_t] - mean_old), axis=0)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(n)))
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(n)
                C_sqrt_inv = np.eye(n)
            
            # Step size adaptation (CSA)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Convergence check: if sigma is very small or no improvement for a while
            if sigma < 1e-10:
                break
            
            # If we're not improving, we might want to restart or adapt
            # For now, just continue with CMA-ES

        return best_x, f_best
# EVOLVE-BLOCK-END
