"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f7 (Step Ellipsoidal) in 5D.
    
    Strategy:
    1. Phase 1: CMA-ES with lambda=10 for global exploration and shape adaptation.
       Includes stagnation detection with adaptive restarts to escape flat plateaus.
       Runs for ~70% of the budget to find the basin of attraction.
    2. Phase 2: (1+1)-ES local refinement with adaptive step size for the remaining budget.
       This exploits the convergence found by CMA-ES to squeeze out the last bits of improvement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # --- CMA-ES Parameters ---
        lam = 10
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Start with a reasonably large step
        
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        # Initial evaluation
        f_mean = func(x_mean)
        evals_used += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        # Phase 1: CMA-ES with stagnation detection
        cma_budget = int(budget * 0.7)
        
        # Stagnation tracking
        last_improvement_eval = 0
        stagnation_threshold = 50  # evaluations without improvement before restart
        restart_count = 0
        max_restarts = 5
        
        while evals_used < cma_budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
                break

            # Generate offspring
            zs = rng.standard_normal((lam, dim))
            ys = zs @ L.T
            xs = x_mean + sigma * ys
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                f_val = func(xs[i])
                evals_used += 1
                fs[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = xs[i].copy()
                    last_improvement_eval = evals_used
                    
            # Sort
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update sigma
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            sum_w_y = np.sum(w[:mu, None, None] * ys[:mu, :, None] * ys[:mu, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails repeatedly, reset C to identity to avoid crash
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    sigma = 1.0
                    x_mean = best_x.copy()
            
            # Stagnation detection and adaptive restart
            if evals_used - last_improvement_eval >= stagnation_threshold and restart_count < max_restarts:
                restart_count += 1
                # Increase sigma to escape plateau
                sigma *= 1.5
                # Reset covariance to identity to explore new directions
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                # Perturb the mean slightly in a random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x_mean = best_x + 0.5 * sigma * direction
                x_mean = np.clip(x_mean, -5, 5)
                last_improvement_eval = evals_used
        
        # Phase 2: Local Refinement (1+1)-ES
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            # Start with a small step size, maybe related to current sigma but smaller
            sigma_local = max(sigma * 0.1, 1e-3)
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                x_new = x_local + sigma_local * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.8
                    
                if sigma_local < 1e-10:
                    sigma_local = 1e-10

        return best_x, best_f
# EVOLVE-BLOCK-END
