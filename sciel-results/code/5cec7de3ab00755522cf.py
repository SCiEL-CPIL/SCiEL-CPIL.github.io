"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and rotation-invariant sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial point: try center first, then a few random points to get a good start
        candidates = [
            np.zeros(d),
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
            rng.uniform(-4, 4, d),
        ]
        
        best_x = None
        best_f = np.inf
        for cand in candidates:
            fc = func(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
        
        used_calls = len(candidates)
        if used_calls >= budget:
            return best_x, best_f

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Step size
        sigma = 1.0
        # Mean
        m = best_x.copy()
        # Covariance matrix (start as identity)
        C = np.eye(d)
        # Inverse square root of C for sampling
        # We'll use Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
        except:
            L = np.eye(d)
            L_inv = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros((d, d))
        
        # Parameters
        mu = lam // 2  # number of selected individuals
        weights = np.ones(mu)
        weights[0] = 2.0  # higher weight for best
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        remaining = budget - used_calls
        eval_count = used_calls

        for _ in range(remaining):
            if eval_count >= budget:
                break
                
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(d)
                # Sample from N(0, C)
                try:
                    y = m + sigma * (L @ z)
                except:
                    y = m + sigma * z
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            f_offspring = []
            for y in offspring:
                fy = func(y)
                f_offspring.append(fy)
                eval_count += 1
                if eval_count >= budget:
                    break
            
            # Pad if we ran out of budget
            while len(f_offspring) < lam:
                f_offspring.append(np.inf)
            
            # Sort by fitness
            indices = np.argsort(f_offspring)
            offspring_sorted = [offspring[i] for i in indices]
            f_sorted = [f_offspring[i] for i in indices]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = offspring_sorted[0].copy()
            
            if eval_count >= budget:
                break

            # Update mean
            new_m = np.zeros(d)
            for i in range(mu):
                new_m += weights[i] * offspring_sorted[i]
            
            # Update sigma
            # Estimate norm of (new_m - m)
            m_diff = new_m - m
            # Normalize by sigma and C
            try:
                m_diff_C_inv = L_inv @ (m_diff / sigma)
            except:
                m_diff_C_inv = m_diff / sigma
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * m_diff_C_inv
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count - used_calls + 1))) < (1.4 + 2 / (d + 1)) * chi_n) else 0.0
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc
            if h_sigma:
                pc += c_c * (2 - c_c) * np.outer(m_diff_C_inv, m_diff_C_inv)
            
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                try:
                    diff = L_inv @ ((offspring_sorted[i] - m) / sigma)
                except:
                    diff = (offspring_sorted[i] - m) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c)) + c_mu - c_1) * C \
                - c_1 * pc \
                - c_mu * rank_mu
            
            # Ensure C is positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = (eigvecs * eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
                L_inv = np.linalg.inv(L)
            except:
                C = np.eye(d)
                L = np.eye(d)
                L_inv = np.eye(d)
            
            m = new_m
            m = np.clip(m, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
