"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) with multi-restart CMA-ES."""
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with fast initial exploration for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def cmaes_run(x0, sigma, max_calls, func, calls_ref, best_x, best_f):
            """Run CMA-ES with small population for fast adaptation."""
            lambda_ = 4
            mu = 2
            w = np.array([1.0, 0.5])
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0
            best_f_local = best_f
            best_x_local = best_x.copy()
            stagnant = 0
            
            while calls_ref[0] + lambda_ <= max_calls and time_ok():
                # Generate population
                ys = []
                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    # Use eigen-based sampling for stability
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.maximum(vals, 1e-10)
                    B = vecs @ np.diag(np.sqrt(vals))
                    y = x0 + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                
                # Evaluate
                fs = []
                for y in ys:
                    fy = func(y)
                    fs.append(fy)
                    calls_ref[0] += 1
                    if calls_ref[0] >= max_calls:
                        break
                
                if calls_ref[0] >= max_calls:
                    break
                
                fs = np.array(fs)
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                fs = fs[order]
                
                if fs[0] < best_f_local:
                    best_f_local = fs[0]
                    best_x_local = ys[0].copy()
                    stagnant = 0
                else:
                    stagnant += 1
                
                # Restart if stagnant for 15 generations or sigma too small
                if stagnant > 15 or sigma < 1e-6:
                    break
                
                x_new = sum(w[i] * ys[i] for i in range(mu))
                
                # Update sigma
                vals, vecs = np.linalg.eigh(C)
                vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
                C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma
                
                hsig = 1.0
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                # Update covariance
                pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
                yk = x_new - x0
                rank1 = np.outer(yk, yk)
                rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))
                
                C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                    + c_1 * (pc @ pc + (1 - (1 - c_c)**2) * C) \
                    + c_mu * rankmu
                C = (C + C.T) / 2
                
                x0 = x_new
                gen += 1
            
            return best_x_local, best_f_local

        # Phase 1: Initial exploration with random points to find promising regions
        calls_ref = [0]
        best_x = np.zeros(dim)
        best_f = np.inf
        
        n_init = min(20, budget // 5)
        for i in range(n_init):
            if not time_ok() or calls_ref[0] >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            calls_ref[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Multi-restart CMA-ES
        # Start from best point, restart if stagnation
        x0 = best_x.copy()
        sigma = 2.0  # Large initial step to explore basin
        
        while calls_ref[0] < budget and time_ok():
            best_x, best_f = cmaes_run(x0, sigma, budget, func, calls_ref, best_x, best_f)
            
            # If we used most of the budget, stop
            if calls_ref[0] >= budget:
                break
            
            # Check if we should restart: if best_f is still not great, try a new random start
            # Lunacek has a local minimum; if we're stuck, try elsewhere
            if best_f > 50 and time_ok() and calls_ref[0] < budget - 20:
                # Try a new random starting point
                x0 = np.random.uniform(-4, 4, dim)
                sigma = 2.0
            else:
                # Continue from current best with slightly smaller sigma
                x0 = best_x.copy()
                sigma = 1.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
