"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # CMA-ES Parameters
        n = self.dim
        # Initial step size: roughly 1/3 of the domain width
        sigma = 1.0
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Weighting for recombination
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State
        x_mean = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        # Initial exploration: evaluate a few random points to get a better starting point
        rng = np.random.default_rng(42)
        n_init = 10
        for _ in range(n_init):
            if time.time() > deadline:
                break
            if self.call_count >= self.budget:
                break
            x_rand = rng.uniform(-5, 5, n)
            wrapped_func(x_rand)
        
        if self.best_x is not None:
            x_mean = self.best_x.copy()
        else:
            x_mean = np.zeros(n)

        # Main CMA-ES Loop
        while self.call_count < self.budget and time.time() < deadline:
            # Generate population
            x_samples = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # x = x_mean + sigma * B @ (D * z)
                x_samples[i] = x_mean + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                x_samples[i] = np.clip(x_samples[i], -5, 5)
            
            # Evaluate population
            f_values = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                if self.call_count >= self.budget:
                    break
                f_values[i] = wrapped_func(x_samples[i])
            
            # Sort by fitness
            idx = np.argsort(f_values)
            x_sorted = x_samples[idx]
            f_sorted = f_values[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + c_c * (sqrt(mu_eff) / sigma) * (x_mean - x_mean_old)
            pc = (1 - c_c) * pc + c_c * (np.sqrt(mu_eff) / sigma) * (x_mean - x_mean_old)
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B.T @ (x_mean - x_mean_old)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (x_mean - x_mean_old))
            
            # Heaviside function
            h_sig = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(-2 * self.call_count / (n * c_sigma))) < (1.4 + 2/(n+1)) * chi_n) else 0
            
            # Update C
            # delta_hsig = (1 - h_sig) * c_sigma * (2 - c_sigma)
            delta_hsig = (1 - h_sig) * c_sigma * (2 - c_sigma)
            
            # Rank-1 update
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            
            # Rank-mu update
            for i in range(mu):
                z_i = (x_sorted[i] - x_mean_old) / sigma
                C += c_mu * np.outer(z_i, z_i)
            
            # Add the sigma * ps * ps.T term if h_sig is 0
            if h_sig == 0:
                C += c_sigma * (2 - c_sigma) * np.outer(ps, ps)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                # Ensure D is positive
                D = np.sqrt(np.maximum(D, 1e-12))
            except np.linalg.LinAlgError:
                # Fallback if eigenvalue decomposition fails
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-12, 10)

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
