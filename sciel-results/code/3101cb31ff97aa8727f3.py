"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with restarts for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        best_x_global = None
        best_f_global = np.inf
        calls_used = 0

        # Strategy: Multi-start with adaptive budget allocation
        # f13 is a sharp ridge. CMA-ES is good but can get stuck.
        # We will run multiple short CMA-ES instances.
        # Each instance gets a portion of the budget.
        # We use a "survival of the fittest" restart strategy.
        
        # Number of restarts
        # If we use too many, we don't explore deep enough.
        # If too few, we might get stuck.
        # Let's try 5 restarts.
        # Budget per restart: budget / 5 = 100 calls.
        # This is small for CMA-ES in dim=5.
        # Maybe fewer restarts, e.g., 3?
        # Or dynamic: if progress is good, continue. If stuck, restart.
        
        # Let's implement a dynamic restart strategy.
        # Start with a larger budget for the first run.
        # If the first run finds a good solution, refine it.
        # If it gets stuck, restart with a new center.

        def run_cma_es(x_start, sigma_start, budget_alloc, x_best, f_best):
            nonlocal calls_used, best_x_global, best_f_global
            
            if calls_used >= budget:
                return x_best, f_best

            # CMA-ES Parameters for dim=5
            lam = 9
            mu = lam // 2  # 4
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Standard CMA-ES parameters
            c1 = (2 / ((dim + 1.3)**2 + mueff)) / 10
            cmu = min(1 - c1, 2 * ((mueff - 2 + 2 / dim) / ((dim + 2)**2 + mueff))) / 10
            cs = (mueff + 2) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            csigma = 1 / ds
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # State
            xmean = np.array(x_start, dtype=float)
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Local best tracking
            x_best_local = np.array(x_best, dtype=float) if x_best is not None else xmean.copy()
            f_best_local = f_best if f_best is not None else np.inf
            
            # Initial evaluation if x_start is new
            if f_best_local == np.inf or np.linalg.norm(xmean - x_best_local) > 1e-6:
                f_start = func(xmean)
                calls_used += 1
                if f_start < f_best_local:
                    f_best_local = f_start
                    x_best_local = xmean.copy()
                    if f_start < best_f_global:
                        best_f_global = f_start
                        best_x_global = xmean.copy()
            else:
                f_start = f_best_local

            # Precompute B and D for sqrt(C)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            
            # Main Loop for this restart
            while calls_used < budget and (budget_alloc > 0 or True): # budget_alloc is just a hint, we break on global budget or time
                if time.time() > deadline:
                    break
                
                # Generate population
                pop_x = np.zeros((lam, dim))
                pop_f = np.zeros(lam)
                
                for i in range(lam):
                    if calls_used >= budget or time.time() > deadline:
                        break
                        
                    z = np.random.standard_normal(dim)
                    y = B @ (D * z)
                    xi = xmean + sigma * y
                    xi = np.clip(xi, -5, 5)
                    pop_x[i] = xi
                    pop_f[i] = func(xi)
                    calls_used += 1
                    
                    if pop_f[i] < f_best_local:
                        f_best_local = pop_f[i]
                        x_best_local = xi.copy()
                        if pop_f[i] < best_f_global:
                            best_f_global = pop_f[i]
                            best_x_global = xi.copy()

                if calls_used >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                idx = np.argsort(pop_f)
                pop_x = pop_x[idx]
                pop_f = pop_f[idx]

                # Update mean
                xmean_old = xmean.copy()
                xmean = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)

                # Update step size control variables
                y_mean = (xmean - xmean_old) / sigma
                
                pc_norm = np.linalg.norm(pc)
                if pc_norm / chi_n < (1.4 + 2 / (dim + 1)) * chi_n:
                    hs = 1.0
                else:
                    hs = 0.0
                    
                pc = (1 - cs) * pc + hs * cs * y_mean
                ps = (1 - csigma) * ps + csigma * hs * y_mean

                # Update C
                delta_y = (pop_x[:mu] - xmean_old) / sigma
                delta_y_minus_mean = delta_y - y_mean
                
                sum_outer = np.sum(weights[:, np.newaxis, np.newaxis] * 
                                   delta_y_minus_mean[:, :, np.newaxis] * 
                                   delta_y_minus_mean[:, np.newaxis, :], 
                                   axis=0)
                
                c_c = (1 + c1 * (2 - c1) - c1 - cmu * np.sum(weights**2))
                C = c_c * C + c1 * (np.outer(pc, pc) + (1 - hs) * c1 * hs * C) + cmu * sum_outer
                C = (C + C.T) / 2

                # Update sigma
                ps_norm = np.linalg.norm(ps)
                sigma = sigma * np.exp(cs / ds * (ps_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update B and D
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))

                # Convergence check
                if sigma < 1e-8 or np.linalg.norm(xmean - xmean_old) < 1e-10:
                    break

            return x_best_local, f_best_local

        # Multi-start Strategy
        # 1. Start at origin (0,0,0,0,0) with sigma=2.0
        # 2. If we use up 50% of budget without significant improvement, restart at a random point.
        # 3. Or simply divide budget into chunks.
        
        # Let's try: 
        # Chunk 1: 60% of budget at origin.
        # Chunk 2: 20% of budget at best found so far (refinement).
        # Chunk 3: 20% of budget at a random point (exploration).
        
        # Actually, for f13 (sharp ridge), the minimum is often not at the origin.
        # A random start is often better.
        # Let's do 3 equal restarts.
        
        num_restarts = 3
        budget_per_restart = budget // num_restarts
        
        # First restart: Origin
        x_start_1 = np.zeros(dim)
        x_best, f_best = run_cma_es(x_start_1, 2.0, budget_per_restart, None, np.inf)
        
        if calls_used >= budget or time.time() > deadline:
            pass
        else:
            # Second restart: Random point
            # Use a point that is not too close to the first start
            x_start_2 = np.random.uniform(-3, 3, dim)
            # If the first run found something good, maybe start near it but with larger sigma?
            # Or just random.
            # Let's start at the best found so far but with a larger sigma to explore out of the local basin?
            # No, CMA-ES shrinks sigma. If we restart at best, we just refine.
            # To explore, we need a new center.
            # Let's pick a random point.
            x_best, f_best = run_cma_es(x_start_2, 2.0, budget_per_restart, x_best, f_best)
            
            if calls_used >= budget or time.time() > deadline:
                pass
            else:
                # Third restart: Another random point
                x_start_3 = np.random.uniform(-3, 3, dim)
                x_best, f_best = run_cma_es(x_start_3, 2.0, budget_per_restart, x_best, f_best)
        
        # Use any remaining budget for final refinement at the global best
        if calls_used < budget and time.time() < deadline and best_x_global is not None:
             # Refine the global best with a small sigma
             # We need to be careful not to exceed budget
             # We can run a short CMA-ES starting at best_x_global with a small sigma
             # But we need to handle the state properly.
             # For simplicity, let's just do a few local steps or a short CMA run.
             # Let's run a short CMA-ES with the remaining budget.
             # We pass the current best as the start.
             # We need to reset the CMA-ES state to start from best_x_global.
             
             # This requires re-initializing the CMA-ES state.
             # The run_cma_es function does this.
             # We just call it again.
             run_cma_es(best_x_global, 0.5, budget - calls_used, best_x_global, best_f_global)

        if best_x_global is None:
            best_x_global = np.zeros(dim)
            best_f_global = func(best_x_global)
            calls_used += 1

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
