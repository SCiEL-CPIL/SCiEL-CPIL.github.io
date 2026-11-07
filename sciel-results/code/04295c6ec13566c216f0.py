"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    """
    Hybrid optimizer: CMA-ES for global search + L-BFGS-B/Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            if call_count >= budget:
                return best_f
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES Global Search ---
        # We allocate ~70% of the budget to CMA-ES to ensure we find the basin.
        cma_budget = int(budget * 0.7)
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # Population size
        mu = lam // 2  # Number of recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + dim / mu_eff) / (dim + 4 + 2 * mu_eff)
        c_1 = 2 / (dim ** 2 + 18)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) ** 2 - 1))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial State
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covering a good portion of the box
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Precompute B and D for initial C
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.clip(eigvals, 1e-12, None))
        B = eigvecs
        
        # Evaluate initial mean
        tracked_func(x_mean)
        
        gen = 0
        # Run CMA-ES until budget or time is exhausted
        while call_count < cma_budget and time.time() < deadline:
            # Check if we should stop early if sigma is too small
            if sigma < 1e-10:
                break
                
            # Generate offspring
            off = []
            off_f = []
            for i in range(lam):
                if call_count >= cma_budget:
                    break
                if time.time() > deadline:
                    break
                z = np.random.default_rng(42 + gen * 100 + i).standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                off.append(y)
                fy = tracked_func(y)
                off_f.append(fy)
            
            if len(off) < mu:
                break
                
            off_f = np.array(off_f)
            idx = np.argsort(off_f)
            off = [off[i] for i in idx[:mu]]
            off_f = off_f[idx[:mu]]
            
            # Recombine
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * off[i]
            
            # Update paths
            x_old = x_mean.copy()
            x_mean = x_new
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_mean - x_old)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (x_mean - x_old)))
            
            # Adapt sigma
            delta_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10)
            
            # Adapt covariance
            # Rank-1 update
            C = (1 - c_1) * C + c_1 * (np.outer(pc, pc) + (1 - delta_sig) * c_sigma * (2 - c_sigma) * C)
            # Rank-mu update
            y_inv = [B @ (D * (x_old - x)) for x in off]
            C += c_mu * sum([w[i] * np.outer(y_inv[i], y_inv[i]) for i in range(mu)])
            
            # Symmetrize and re-decompose
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            gen += 1

        # --- Phase 2: Local Refinement ---
        # Use L-BFGS-B from the best point found by CMA-ES
        if call_count < budget - 10 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    tracked_func,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 1000,
                        'maxfun': budget - call_count - 5,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        # --- Phase 3: Nelder-Mead Polish ---
        # If we have budget left, do a short NM run to polish
        if call_count < budget - 5 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    tracked_func,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': budget - call_count,
                        'adaptive': True,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
