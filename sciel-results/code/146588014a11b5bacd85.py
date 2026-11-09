"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for rotated Rastrigin in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s

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

        # CMA-ES implementation
        def cma_es(start_x, max_calls, initial_sigma=2.0, lam=None):
            nonlocal best_x, best_f
            
            if lam is None:
                lam = max(10, int(4 + 3 * np.log(dim)))
            
            x = np.clip(start_x, -5, 5).copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = initial_sigma
            used = 1
            
            # CMA-ES parameters
            mu = lam // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Sort weights for selection
            w_sorted = np.sort(weights)[::-1]
            
            while used < max_calls and call_count < budget and time.time() < deadline:
                # Generate population
                pop = []
                for i in range(lam):
                    y = x + sigma * C_sqrt @ rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    pop.append(y)
                
                # Evaluate population
                fvals = []
                for y in pop:
                    fy = safe_func(y)
                    fvals.append(fy)
                    used += 1
                    if call_count >= budget or time.time() > deadline:
                        break
                
                if len(fvals) < lam:
                    # Pad with inf if we ran out of budget
                    while len(fvals) < lam:
                        fvals.append(np.inf)
                
                # Sort by fitness
                sorted_idx = np.argsort(fvals)
                selected = [pop[i] for i in sorted_idx[:mu]]
                selected_f = [fvals[i] for i in sorted_idx[:mu]]
                
                # Update best
                if selected_f[0] < best_f:
                    best_f = selected_f[0]
                    best_x = selected[0].copy()
                
                # Compute mean of selected
                x_new = np.sum([w * xi for w, xi in zip(weights, selected)], axis=0)
                
                # Update step size (sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (x_new - x))
                if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / dim) * np.sqrt((1 - c_sigma) / (1 - c_sigma ** (2 * (used - 1)))):
                    pass  # No hessian correction needed for simplicity
                sigma = sigma * np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Update covariance matrix
                x_old = x
                x = x_new
                
                # Compute rank-one and rank-mu updates
                diff = x - x_old
                p_c = (1 - c_c) * p_c + c_c * diff / sigma
                delta_heav = (1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (used - 1))) / chi_n < (1.4 + 1 / dim) else 0)
                
                C = (1 + c_1 * delta_heav - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) + (delta_heav - 1) * C) \
                    + c_mu / sigma ** 2 * np.sum([w * np.outer(xi - x_old, xi - x_old) for w, xi in zip(weights, selected)], axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update C_sqrt (simplified: recompute via eigendecomposition)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_sqrt = np.eye(dim)
        
        # Phase 1: Single CMA-ES run with large population for global exploration
        cma_es(np.zeros(dim), budget, initial_sigma=2.0, lam=12)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
