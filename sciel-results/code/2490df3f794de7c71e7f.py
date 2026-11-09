"""Rewritten optimizer: CMA-ES with restarts for Rastrigin f3."""
import time
import numpy as np


class Optimizer:
    """CMA-ES based optimizer with restarts for Rastrigin-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        dim = self.dim
        budget = self.budget
        self.calls = 0

        def _eval(x):
            if self.calls >= budget:
                return np.inf
            self.calls += 1
            return float(func(x))

        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))
        pop_size = max(pop_size, 6)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Phase 1: CMA-ES with restarts
        # Use most of the budget for CMA-ES
        n_cma_calls = int(budget * 0.85)
        
        def run_cma_es(x0, sigma0, max_calls, rng_local):
            """Run CMA-ES starting from x0 with initial sigma0."""
            nonlocal best_x, best_f
            
            n = dim
            # CMA-ES parameters
            mu = max(1, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Step size
            sigma = sigma0
            
            # Mean
            mean = x0.copy()
            
            # Covariance matrix
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            C_sqrt = np.eye(n)
            
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            def sample_population():
                pop = []
                for i in range(pop_size):
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    pop.append(x)
                return pop
            
            def update_covariance(pop, fitnesses):
                """Update covariance matrix and step size."""
                nonlocal sigma, mean, C, C_inv_sqrt, C_sqrt, p_sigma, p_c
                
                # Sort by fitness
                order = np.argsort(fitnesses)
                sorted_pop = [pop[i] for i in order]
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * sorted_pop[i]
                
                # Update evolution paths
                # p_sigma
                sigma_old = sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                          (C_inv_sqrt @ (new_mean - mean) / sigma_old)
                
                # h_sigma check
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max_calls))) < \
                          (1.4 + 2 / (n + 1)) * chi_n
                
                # p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                      (new_mean - mean) / sigma_old
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (sorted_pop[i] - mean) / sigma_old
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                mean = new_mean
                
                # Recompute C_sqrt and C_inv_sqrt
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                return mean, sigma
            
            # Main loop
            local_calls = 0
            no_improve = 0
            restart_threshold = 150
            
            for gen in range(10000):
                if self.calls >= budget or local_calls >= max_calls:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Sample population
                pop = sample_population()
                
                # Evaluate
                fitnesses = []
                for x in pop:
                    f = _eval(x)
                    fitnesses.append(f)
                    local_calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                
                if local_calls >= max_calls:
                    break
                
                # Check for restart
                if no_improve >= restart_threshold:
                    break
                
                # Update
                mean, sigma = update_covariance(pop, fitnesses)
            
            return best_x, best_f
        
        # Run multiple CMA-ES restarts
        n_restarts = 3
        calls_per_restart = n_cma_calls // n_restarts
        
        for r in range(n_restarts):
            if self.calls >= budget or time.time() > deadline - 10.0:
                break
            
            if r == 0:
                # First restart: start from center
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Subsequent restarts: start from best found or random
                if rng.random() < 0.5:
                    x0 = best_x.copy()
                else:
                    x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.5
            
            run_cma_es(x0, sigma0, calls_per_restart, rng)
        
        # Phase 2: Nelder-Mead polish
        remaining_budget = budget - self.calls
        if remaining_budget > (dim + 1) and time.time() < deadline - 5.0:
            # Build initial simplex around best_x
            simplex = [best_x.copy()]
            for d in range(dim):
                perturbed = best_x.copy()
                step = 0.05
                if perturbed[d] + step > 5.0:
                    step = -step
                if perturbed[d] + step < -5.0:
                    step = -step
                perturbed[d] += step
                simplex.append(perturbed)
            
            fvals = []
            for s in simplex:
                if self.calls >= budget:
                    break
                fvals.append(_eval(s))
            
            if len(fvals) < dim + 1:
                return best_x, best_f
            
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            
            while self.calls < budget and time.time() < deadline - 2.0:
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                
                if self.calls >= budget:
                    break
                fr = _eval(xr)
                
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    if self.calls >= budget:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = _eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    if self.calls >= budget:
                        break
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = _eval(xc)
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            if self.calls >= budget:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = _eval(simplex[i])
        
        return best_x, best_f
