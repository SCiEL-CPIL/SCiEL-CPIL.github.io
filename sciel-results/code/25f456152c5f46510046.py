"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Improved CMA-ES with adaptive parameters and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: Global exploration with diverse initial points
        n_initial = min(10, budget // 50)
        candidates = [np.zeros(d)]
        for _ in range(n_initial - 1):
            candidates.append(rng.uniform(-4, 4, d))
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        for cand in candidates:
            fc = func(cand)
            eval_count += 1
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
        
        if eval_count >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES with improved parameters
        # Adaptive population size
        lam = max(4, int(4 + 3 * np.log(d)))
        sigma = 1.5  # Start with larger step for better exploration
        m = best_x.copy()
        C = np.eye(d)
        
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
        mu = lam // 2
        weights = np.ones(mu)
        weights[0] = 2.0
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        remaining = budget - eval_count
        stall_counter = 0
        best_f_history = [best_f]
        stall_threshold = 20  # Number of generations without improvement before considering restart
        restart_sigma = 1.0

        for gen in range(remaining // lam + 1):
            if eval_count >= budget:
                break
            
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(d)
                try:
                    y = m + sigma * (L @ z)
                except:
                    y = m + sigma * z
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            f_offspring = []
            for y in offspring:
                if eval_count >= budget:
                    f_offspring.append(np.inf)
                    continue
                fy = func(y)
                f_offspring.append(fy)
                eval_count += 1
            
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
                stall_counter = 0
                best_f_history.append(best_f)
            else:
                stall_counter += 1
            
            if eval_count >= budget:
                break

            # Check for stall and consider restart
            if stall_counter > stall_threshold:
                # Try a smaller sigma to explore locally
                sigma = max(0.1, sigma * 0.5)
                stall_counter = 0
                # If still stuck, try a new random point
                if sigma < 0.1 and eval_count < budget - 10:
                    new_x = best_x + rng.uniform(-1, 1, d)
                    new_x = np.clip(new_x, -5, 5)
                    new_f = func(new_x)
                    eval_count += 1
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()
                        m = best_x.copy()
                        sigma = restart_sigma
                    else:
                        # Keep exploring from current best
                        pass

            # Update mean
            new_m = np.zeros(d)
            for i in range(mu):
                new_m += weights[i] * offspring_sorted[i]
            
            # Update sigma
            m_diff = new_m - m
            try:
                m_diff_C_inv = L_inv @ (m_diff / sigma)
            except:
                m_diff_C_inv = m_diff / sigma
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * m_diff_C_inv
            
            # h_sigma check
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            if denom < 1e-10:
                denom = 1e-10
            h_sigma = 1.0 if (norm_ps / denom < (1.4 + 2 / (d + 1)) * chi_n) else 0.0
            
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
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
            
            # Ensure C is positive definite with better regularization
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                # Add small regularization for numerical stability
                C += 1e-12 * np.eye(d)
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
            except:
                C = np.eye(d)
                L = np.eye(d)
                L_inv = np.eye(d)
            
            m = new_m
            m = np.clip(m, -5, 5)

        # Phase 3: Local refinement with Nelder-Mead if we have budget left
        if eval_count < budget - 5 and best_x is not None:
            def wrapper(x):
                nonlocal eval_count
                if eval_count >= budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                eval_count += 1
                return fx
            
            try:
                result = minimize(wrapper, best_x, method='Nelder-Mead',
                                options={'maxiter': budget - eval_count, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
