"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Attractive Sector."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        d = self.dim
        n = self.dim

        # Initial population: center + random
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # Initialize CMA-ES parameters
        mean = x0.copy()
        sigma = 2.0
        C = np.eye(d)
        Csqrt = np.eye(d)
        
        # CMA-ES parameters
        mu = max(1, int(np.floor(n / 2)))  # number of selected parents
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Cumulation vectors
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Track best
        best_x = mean.copy()
        best_f = f0
        calls_made = 1
        
        # Initial population
        pop = [mean.copy()]
        pop_f = [f0]
        
        # Generate initial population
        for _ in range(n):
            x_new = mean + sigma * Csqrt @ rng.standard_normal(d)
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            pop.append(x_new)
            pop_f.append(f_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        
        # Sort population by fitness
        idx = np.argsort(pop_f)
        pop = [pop[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Update mean
            old_mean = mean.copy()
            mean = sum(w[i] * pop[i] for i in range(mu))
            
            # Update step size control vector ps
            csqrt = np.sqrt(1 - (1 - c_sigma) ** 2 / (n + 1))
            # Inverse of Csqrt (approximate using C since C is close to identity initially)
            # For simplicity and speed, use the standard CMA-ES update
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * Csqrt_inv @ (mean - old_mean)
            # Since we don't have explicit Csqrt_inv, we approximate. 
            # Actually, Csqrt @ Csqrt = C. So Csqrt_inv = Csqrt.T @ C_inv. 
            # This is getting complex. Let's use the standard simplified CMA-ES update.
            
            # Standard CMA-ES update for ps
            # We need C^{-1/2}. Let's approximate with C^{-1} for the direction, or just use the mean difference.
            # A simpler, robust CMA-ES implementation:
            
            # Recalculate Csqrt and its inverse
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                Cinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                Cinv = np.eye(d)
                Cinv_sqrt = np.eye(d)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * Cinv_sqrt @ (mean - old_mean) / sigma
            
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - n))) / np.sqrt(n)
            hsig = (hs < (1 + 1.4 / n) * np.sqrt((n + 1) * (1 - (1 - c_sigma) ** (2 * (calls_made - n)))))
            
            # Update covariance matrix
            if hsig:
                pc = (1 - c_c) * pc + c_c * (mean - old_mean) / sigma
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * sum(w[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean) for i in range(mu)) / sigma ** 2
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(d)
            
            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Generate new population
            new_pop = []
            new_pop_f = []
            
            # Update Csqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                Csqrt = np.eye(d)
            
            for _ in range(n):
                x_new = mean + sigma * Csqrt @ rng.standard_normal(d)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                new_pop.append(x_new)
                new_pop_f.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            # Sort new population
            idx = np.argsort(new_pop_f)
            pop = [new_pop[i] for i in idx]
            pop_f = [new_pop_f[i] for i in idx]
            
            # Check for convergence
            if sigma < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
