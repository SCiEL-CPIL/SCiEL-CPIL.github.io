"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget or time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return float(fx)
            except Exception:
                return np.inf

        # Phase 1: Coarse random search to identify basin of attraction
        # Use ~20% of budget for initial exploration
        phase1_budget = max(50, budget // 5)
        
        for _ in range(phase1_budget):
            if call_count >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, d)
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # Phase 2: CMA-ES implementation for refinement
        if call_count < budget and time.time() < deadline:
            try:
                # Simple CMA-ES implementation
                sigma = 2.0
                pop_size = max(4, int(4 + 3 * np.log(d)))
                best_cma_x = best_x.copy()
                best_cma_f = best_f
                mean = best_x.copy()
                C = np.eye(d)
                sigma = 1.5
                chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
                c_sigma = (mu_eff + 0.3) / (d + 2.5)
                c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
                mu = pop_size // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights**2)
                step_size = sigma
                c1 = 2 / ((d + 1.3)**2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + mu_eff))
                cs = (mu_eff + 2) / (d + mu_eff + 5)
                ds = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + cs
                pc = np.zeros(d)
                ps = np.zeros(d)
                c_sigma = (mu_eff + 0.3) / (d + 2.5)
                c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
                chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
                
                max_iter = (budget - call_count) // pop_size
                for it in range(max_iter):
                    if time.time() > deadline or call_count + pop_size > budget:
                        break
                    
                    # Generate population
                    B = rng.standard_normal((pop_size, d))
                    # Whitening transform using Cholesky
                    L = np.linalg.cholesky(C)
                    B = B @ L.T
                    pop = np.array([mean + sigma * b for b in B])
                    
                    # Evaluate population
                    fvals = []
                    for i in range(pop_size):
                        fx = wrapped_func(pop[i])
                        fvals.append(fx)
                        if call_count >= budget or time.time() > deadline:
                            break
                    
                    if len(fvals) < pop_size:
                        break
                    
                    fvals = np.array(fvals)
                    # Sort by fitness
                    order = np.argsort(fvals)
                    pop_sorted = pop[order]
                    fvals_sorted = fvals[order]
                    
                    # Update best
                    if fvals_sorted[0] < best_cma_f:
                        best_cma_f = fvals_sorted[0]
                        best_cma_x = pop_sorted[0].copy()
                        if best_cma_f < best_f:
                            best_f = best_cma_f
                            best_x = best_cma_x.copy()
                    
                    # Update mean
                    new_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
                    
                    # Update covariance matrix
                    y = (new_mean - mean) / sigma
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y
                    hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**((it + 1) / ds)) < (1.4 + 2 / (d + 1)) * chi_d)
                    ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                    sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps)**2 / chi_d - 1))
                    sigma = max(sigma, 1e-12)
                    
                    delta_sig = (1 - hsig) / c_sigma + 1
                    C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_sig - 1) * C)
                    for i in range(mu):
                        yi = (pop_sorted[i] - mean) / sigma
                        C += cmu * weights[i] * np.outer(yi, yi)
                    mean = new_mean
                    
                    # Clip mean to stay in bounds
                    mean = np.clip(mean, -5, 5)
                    
                    # Check convergence
                    if sigma < 1e-8 or it > 0 and np.all(np.abs(pop_sorted[:mu] - mean) < 1e-6):
                        break
                        
                if best_cma_f < best_f:
                    best_f = best_cma_f
                    best_x = best_cma_x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
