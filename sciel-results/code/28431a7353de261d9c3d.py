"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid Global Search + CMA-ES for Rosenbrock minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Phase 1: Global Exploration
        n_explore = min(50, self.budget // 10)
        if n_explore == 0:
            n_explore = 1
            
        x_best = None
        f_best = np.inf
        
        for _ in range(n_explore):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
                
        if x_best is None:
            x_best = rng.uniform(-5, 5, self.dim)
            f_best = func(x_best)
            n_explore += 1

        # Phase 2: CMA-ES Refinement
        budget_left = self.budget - n_explore
        if budget_left <= 0:
            return x_best, f_best

        N = self.dim
        lam = max(4, int(4 + 3 * np.log(N)))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (N + mu_eff + 5)
        c_c = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        c_1 = 2 / ((N + 1.3) ** 2 + mu_eff)
        c_mu = min(c_1 / 2, 2 * (mu_eff / mu - 2) / ((N + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 2) / (mu_eff + 2)) - 1) + c_sigma
        
        x_m = x_best.copy()
        sigma = 1.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        B = np.linalg.cholesky(C)
        
        calls_used = n_explore
        best_x = x_best
        best_f = f_best
        t = 0
        
        while calls_used < self.budget:
            pop = np.zeros((lam, N))
            for i in range(lam):
                z = rng.standard_normal(N)
                pop[i] = x_m + sigma * (B @ z)
            
            n_eval = min(lam, self.budget - calls_used)
            if n_eval == 0:
                break
                
            fvals = np.zeros(n_eval)
            for i in range(n_eval):
                fvals[i] = func(pop[i])
                calls_used += 1
                
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]
            fvals_sorted = fvals[idx]
            
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
                
            x_m_new = np.sum(weights[:mu] * pop_sorted[:mu], axis=0)
            
            y = (x_m_new - x_m) / sigma
            # B_inv y
            B_inv_y = np.linalg.solve(B, y)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B_inv_y
            
            t += 1
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * t)) <
                     (1 + 2 / (N + 1)) * np.sqrt((N + 1) / (N - 1)))
                     
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            pc = (1 - c_c) * pc + h_sig * c_c * y
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(y, y) + (1 - h_sig * c_c * (2 - c_c)) * np.outer(pc, pc))
            
            for i in range(mu):
                # zi = B^{-1} (xi - x_m_old) / sigma
                # But we have pop_sorted[i] which is the new point.
                # The update uses the step from the old mean.
                # zi = B^{-1} (pop_sorted[i] - x_m) / sigma
                z_i = np.linalg.solve(B, (pop_sorted[i] - x_m) / sigma)
                C += c_mu * weights[i] * np.outer(z_i, z_i)
                
            x_m = x_m_new
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(N)
                B = np.linalg.cholesky(C)
                
        return best_x, best_f

# EVOLVE-BLOCK-END
