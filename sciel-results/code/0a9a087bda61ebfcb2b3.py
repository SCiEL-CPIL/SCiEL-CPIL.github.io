"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with stagnation detection for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0

        calls_made = 0
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return best_f
            if time.time() > deadline:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter, budget_limit):
            nonlocal calls_made, best_f, best_x, deadline
            n = dim
            popsize = 4 + int(3 * np.log(n))
            lambda_ = popsize
            mu = lambda_ // 2
            
            # CMA-ES parameters
            cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
            cs = (n + 1.3) / (n + 3 + mu_eff) if (mu_eff := mu * (1 - 1.0 / mu)) > 0 else 1.0
            c1 = 2 / ((n + 1.8) ** 2 + mu)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Initialize
            mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial population
            z = rng.standard_normal((lambda_, n))
            pop = mean + sigma * (z @ (B * D))
            pop = np.clip(pop, -5.0, 5.0)
            
            # Evaluate
            fvals = np.array([safe_func(p) for p in pop])
            
            # Sort
            order = np.argsort(fvals)
            y_sorted = z[order]
            f_sorted = fvals[order]
            x_sorted = pop[order]
            
            # Weighted recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size
            D_old = D.copy()
            sigma_old = sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D @ (B.T @ (mean_old - mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made + 1))) < (1 + 2 / (n + 1)) * np.sqrt(n + 1))
            sigma *= np.exp(cs / n * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update covariance
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) / D @ (B.T @ (mean_old - mean))
            else:
                pc = (1 - cc) * pc
            
            delta = (2 * int(hsig) - 1) * cc * (2 - cc)
            C = (1 + c1 * delta - c1 - cmu) * np.eye(n)
            for i in range(mu):
                yi = (B * D) @ (mean_old - x_sorted[i]) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)
            
            # Eigen decomposition for B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Stagnation detection
            stagnation_counter = 0
            prev_best = best_f
            for iteration in range(1, max_iter):
                if time.time() > deadline or calls_made >= budget_limit:
                    break
                
                # Generate new population
                z = rng.standard_normal((lambda_, n))
                pop = mean + sigma * (z @ (B * D))
                pop = np.clip(pop, -5.0, 5.0)
                
                # Evaluate
                fvals = np.array([safe_func(p) for p in pop])
                
                # Sort
                order = np.argsort(fvals)
                y_sorted = z[order]
                f_sorted = fvals[order]
                x_sorted = pop[order]
                
                # Update mean
                mean_old = mean.copy()
                mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / D @ (B.T @ (mean_old - mean))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made + 1))) < (1 + 2 / (n + 1)) * np.sqrt(n + 1))
                sigma *= np.exp(cs / n * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Update covariance
                if hsig:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) / D @ (B.T @ (mean_old - mean))
                else:
                    pc = (1 - cc) * pc
                
                delta = (2 * int(hsig) - 1) * cc * (2 - cc)
                C = (1 + c1 * delta - c1 - cmu) * np.eye(n)
                for i in range(mu):
                    yi = (B * D) @ (mean_old - x_sorted[i]) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)
                
                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                # Check for stagnation
                if best_f < prev_best - 1e-10:
                    prev_best = best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # If stuck, restart around best point with smaller sigma
                if stagnation_counter > 15:
                    break
            
            return mean, sigma

        # Phase 1: CMA-ES from multiple starting points
        n_starts = 3
        start_points = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-3.0, 3.0, dim))
        
        for x0 in start_points:
            if time.time() > deadline or calls_made >= total_budget:
                break
            
            # Run CMA-ES with limited budget per start
            remaining = total_budget - calls_made
            max_iter = min(50, remaining // 10)  # Approximate iterations
            if max_iter < 5:
                continue
            
            _, _ = cma_es(x0, 0.5, max_iter, total_budget)
        
        # Phase 2: Fine-tuning with small perturbations around best
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_fine = 0.05
            
            for _ in range(min(30, total_budget - calls_made)):
                if time.time() > deadline or calls_made >= total_budget:
                    break
                
                y = x + sigma_fine * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_fine *= 1.1
                else:
                    sigma_fine *= 0.9
        
        return best_x, best_f

# EVOLVE-BLOCK-END
