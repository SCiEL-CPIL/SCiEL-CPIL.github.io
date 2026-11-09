"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based black-box continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def eval_func(x):
            nonlocal calls_used
            calls_used += 1
            val = func(x)
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES Parameters
        lam = max(4, 4 + int(3 * np.log(d)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / d) / (d + 2 + mu_eff) - 1))
        
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d*d))

        # State
        mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Initial evaluation
        f_mean = eval_func(mean)
        update_best(mean, f_mean)

        # Main Loop
        while calls_used + lam <= budget:
            # Sample
            pop = np.empty((lam, d))
            f_pop = np.empty(lam)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += np.eye(d) * 1e-10
                L = np.linalg.cholesky(C)
            
            for i in range(lam):
                z = rng.standard_normal(d)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop[i] = x
                f_pop[i] = eval_func(x)
                update_best(x, f_pop[i])
            
            # Select
            idx = np.argsort(f_pop)
            pop_sel = pop[idx[:mu]]
            
            # Update Mean
            mean_new = np.sum(weights[:, None] * pop_sel, axis=0)
            
            # Update Paths
            y = (mean_new - mean) / sigma
            pc = (1 - c_c) * pc + c_c * y
            
            # C^-0.5
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y)
            
            # Update Covariance
            h_sigma = (np.sum(ps ** 2) / (1 - (1 - c_sigma) ** (2 * (calls_used + lam)))) < (1 + 2 / (d + 1)) * c_sigma
            h_sigma = float(h_sigma)
            
            rank_1 = np.outer(pc, pc)
            if h_sigma:
                rank_mu = np.zeros((d, d))
            else:
                diff = (pop_sel - mean) / sigma
                rank_mu = np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_1 + (1 - h_sigma) * c_c * (2 - c_c) * rank_1) + c_mu * rank_mu
            
            # Update Sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            mean = mean_new
            
            # Restart if stagnated
            if sigma < 1e-4:
                mean = best_x.copy()
                sigma = 1.0
                C = np.eye(d)
                ps = np.zeros(d)
                pc = np.zeros(d)

        return best_x, best_f
# EVOLVE-BLOCK-END
