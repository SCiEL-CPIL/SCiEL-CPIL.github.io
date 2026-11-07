"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def update_best(x, f):
            nonlocal best_x_global, best_f_global
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()

        def cma_es(x0, f0, budget_alloc):
            """
            Lightweight CMA-ES implementation.
            x0: initial mean (np.array)
            f0: function value at x0
            budget_alloc: max function calls for this run
            """
            nonlocal calls, best_x_global, best_f_global
            
            dim = self.dim
            # Population size rule of thumb: 4 + floor(3 ln n)
            lam = max(4 + int(3 * np.log(dim)), 4)
            
            # Initial sigma
            # For sharp ridge, we want to explore a bit broadly initially
            sigma = 1.0 
            
            # Covariance matrix
            C = np.eye(dim)
            # Eigenvalue decomposition for sampling
            eigval, eigvec = np.linalg.eigh(C)
            
            # Step size control (CCS)
            c_sigma = (mu_eff + 2) / (dim + 4) / (mu_eff + 2) / 2.5
            d_sigma = 1 + 2 * max(0, sqrt(dim - 1) / (dim + 1) - 1) + c_sigma
            sigma = 1.0
            
            # Mean update
            c_mean = 1.0 / (dim * 2) # Roughly
            c_c = (4 + dim / dim) / (dim + 4)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Initial population
            # We already have x0, f0. We need lam-1 more points.
            # To save calls, we can just start with x0 as the first point.
            # But standard CMA-ES samples lam points.
            # Let's sample lam points around x0.
            # Actually, to be efficient, let's just run standard CMA-ES from x0.
            
            # If we have a specific f0, we can use it, but CMA-ES usually evaluates all.
            # Let's just evaluate lam points.
            
            # Check budget
            if budget_alloc < lam + 1:
                return

            # Generate first population
            pop = []
            fpop = []
            for i in range(lam):
                if not time_ok() or calls >= self.budget - 2:
                    break
                z = rng.standard_normal(dim)
                # Sample y = B * D * z
                y = eigvec @ (eigval ** 0.5 * z)
                x = x0 + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                pop.append(x)
                fpop.append(f)
                update_best(x, f)
                
            if len(pop) < lam:
                return
                
            pop = np.array(pop)
            fpop = np.array(fpop)
            
            # Sort
            order = np.argsort(fpop)
            pop = pop[order]
            fpop = fpop[order]
            
            # Initial mean update
            m = x0.copy()
            # Actually, standard CMA-ES starts with m=x0 and updates.
            # Let's update m now.
            m = np.sum(weights[:, None] * pop[:mu], axis=0)
            # Wait, the first point x0 was not necessarily in the population generated above.
            # The population generated above is centered at x0.
            # So m is updated based on the new population.
            
            # Main loop
            while time_ok() and calls < self.budget - 2:
                # Update mean
                m_new = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # Update sigma (CSA)
                z_mean = (m_new - m) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(z_mean) * z_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                
                # Update covariance matrix (CS)
                pc = (1 - c_c) * pc + c_c * z_mean
                v = c_1 * (1 + (2 - c_c) * c_c) * np.outer(pc, pc) + c_mu * np.sum(weights ** 2) * np.outer(pc, pc)
                # Correct formula:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc*pc^T + (1 - (1-c_c)^2) * I) + c_mu * sum(wi * yi*yi^T)
                # Let's stick to a simplified robust update to avoid bugs.
                # Actually, let's just use a simple diagonal adaptation or skip covariance update 
                # if it's too complex for a quick implementation? 
                # No, CMA-ES power is in covariance.
                
                # Let's re-implement the standard update carefully.
                # y_i = x_i - m_old
                y = pop[:mu] - m
                # Normalize y by sigma
                y_norm = y / sigma
                
                # Update pc
                pc = (1 - c_c) * pc + c_c * np.sum(weights[:, None] * y_norm, axis=0)
                
                # Update C
                # Rank-1 update
                rank1 = np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * np.eye(dim)
                # Rank-mu update
                rankmu = np.sum(weights[:, None, None] * y_norm[:, :, None] * y_norm[:, None, :], axis=0)
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                
                # Ensure positive semi-definite
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                C = eigvec @ np.diag(eigval) @ eigvec.T
                
                # Update mean
                m = m_new
                
                # Generate new population
                new_pop = []
                new_fpop = []
                for i in range(lam):
                    if not time_ok() or calls >= self.budget - 2:
                        break
                    z = rng.standard_normal(dim)
                    y_sample = eigvec @ (eigval ** 0.5 * z)
                    x = m + sigma * y_sample
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    calls += 1
                    new_pop.append(x)
                    new_fpop.append(f)
                    update_best(x, f)
                    
                if len(new_pop) < lam:
                    break
                    
                pop = np.array(new_pop)
                fpop = np.array(new_fpop)
                
                # Sort
                order = np.argsort(fpop)
                pop = pop[order]
                fpop = fpop[order]

        # Multi-start CMA-ES
        # 1. Generate a few initial points to find a good basin
        n_initial = 10
        initial_points = []
        for _ in range(n_initial):
            x = rng.uniform(-4, 4, self.dim)
            initial_points.append(x)
        
        # Evaluate initial points
        f_initial = []
        for x in initial_points:
            if not time_ok() or calls >= self.budget - 10:
                break
            f = func(x)
            calls += 1
            f_initial.append(f)
            update_best(x, f)
            
        if not f_initial:
            return best_x_global, best_f_global
            
        # Sort initial points
        sorted_idx = np.argsort(f_initial)
        
        # Run CMA-ES from the best 2-3 initial points
        # Allocate budget: ~400 calls total for CMA-ES
        # If 2 starts, 200 each.
        n_cma_starts = 2
        remaining_budget = self.budget - calls
        alloc_per_start = remaining_budget // n_cma_starts
        
        for i in range(n_cma_starts):
            if i >= len(sorted_idx):
                break
            idx = sorted_idx[i]
            x_start = initial_points[idx].copy()
            f_start = f_initial[idx]
            
            cma_es(x_start, f_start, alloc_per_start)
            
            if not time_ok() or calls >= self.budget - 5:
                break
                
        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
