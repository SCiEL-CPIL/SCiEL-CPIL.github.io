"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage hybrid: CMA-ES for global exploration, then (1+1)-ES polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def run_cma_es(x0, sigma0, max_calls):
            """Run CMA-ES from given start with given budget."""
            nonlocal best_x, best_f, call_count
            
            lam = 12  # Population size
            mu = lam // 2  # Number of parents
            
            # CMA-ES parameters
            c_sigma = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / ((dim + 2) + lam)
            c_c = (4 + dim / lam) / ((dim + 2) ** 2 + lam)
            c_1 = (3 - dim + 2 * c_c) / ((dim + 2) ** 2 + lam)
            c_mu = 2 * (mu - 1 + 1.0 / mu) / ((dim + 2) ** 2 + lam)
            
            # Weights
            w = np.zeros(lam)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cholC = np.linalg.cholesky(C)
            
            # Initial population
            pop = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                xi = x_mean + sigma * (cholC @ rng.standard_normal(dim))
                xi = np.clip(xi, -5, 5)
                fi = safe_func(xi)
                pop.append((fi, xi))
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            if not pop:
                return
            
            pop.sort(key=lambda t: t[0])
            
            # Main loop
            while call_count < budget and time.time() < deadline:
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum([w[i] * pop[i][1] for i in range(mu)], axis=0)
                
                # Update step size control variables
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (cholC.T @ (x_mean - x_mean_old)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count // lam + 1))) <
                        (1.4 + 2 / (dim + 1)) * np.trace(C) / dim)
                
                # Update covariance matrix
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_mean_old) / sigma
                rank1 = np.outer(pc, pc)
                delta_h = (1.0 if hsig else 0.0) * (2 - c_c)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (pop[i][1] - x_mean_old) / sigma
                    rankmu += w[i] * np.outer(zi, zi)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                
                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2.0
                try:
                    cholC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-4 * np.eye(dim)
                    C = (C + C.T) / 2.0
                    cholC = np.linalg.cholesky(C)
                
                # Update sigma
                sigma *= np.exp(c_sigma / (1 + c_sigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                # Generate new population
                pop = []
                for i in range(lam):
                    if call_count >= budget or time.time() > deadline:
                        break
                    xi = x_mean + sigma * (cholC @ rng.standard_normal(dim))
                    xi = np.clip(xi, -5, 5)
                    fi = safe_func(xi)
                    pop.append((fi, xi))
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if not pop or call_count >= budget or time.time() > deadline:
                    break
                
                pop.sort(key=lambda t: t[0])
            
            return x_mean, sigma

        def run_es_polish(start_x, max_calls):
            """Run (1+1)-ES polishing from given start."""
            nonlocal best_x, best_f, call_count
            
            x = start_x.copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 0.5
            used = 1
            
            while used < max_calls and call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break
            
            return x, fx

        # Phase 1: CMA-ES for global exploration (60% of budget)
        if call_count < budget and time.time() < deadline:
            x0 = np.zeros(dim)
            sigma0 = 2.0
            budget1 = int(budget * 0.6)
            run_cma_es(x0, sigma0, budget1)
        
        # Phase 2: Polish the best solution found (40% of budget)
        if call_count < budget and time.time() < deadline:
            if best_x is not None:
                start3 = best_x.copy()
            else:
                start3 = rng.uniform(-5, 5, dim)
            budget3 = budget - call_count
            run_es_polish(start3, budget3)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
