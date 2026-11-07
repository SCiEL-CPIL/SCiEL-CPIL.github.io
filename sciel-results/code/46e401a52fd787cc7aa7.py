"""Multi-start CMA-ES with restart for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_f = np.inf
        evals_used = 0
        
        def time_left():
            return timeout - (time.time() - start_time)
        
        def evaluate(x):
            nonlocal evals_used, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation as a helper
        def run_cma_es(m0, sigma0, max_evals, max_time):
            nonlocal evals_used, best_f, best_x
            nonlocal time_left
            
            # Parameters for d=5
            lam = 10
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
            chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d*d))
            
            m = m0.copy()
            sigma = sigma0
            pc = np.zeros(d)
            C = np.eye(d)
            B = np.eye(d)
            D = np.ones(d)
            
            local_best_f = np.inf
            local_best_x = m.copy()
            
            # Evaluate initial point if not already done? 
            # We assume m0 is fresh or we just start the loop.
            # For multi-start, we should evaluate m0 first.
            f0 = evaluate(m)
            if f0 < local_best_f:
                local_best_f = f0
                local_best_x = m.copy()
            
            evals_in_cma = 1
            last_improvement_eval = 1
            stagnation_threshold = 3 * lam
            
            while evals_in_cma < max_evals and time_left() > 1.0:
                # If remaining budget is too small for a full generation, do (1+1) steps
                if max_evals - evals_in_cma < lam:
                    while evals_in_cma < max_evals and time_left() > 0.5:
                        y = m + sigma * np.random.randn(d)
                        fy = evaluate(y)
                        evals_in_cma += 1
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                            sigma *= 1.05
                        else:
                            sigma *= 0.95
                            if sigma < 1e-4:
                                break
                    break
                
                # Generate population
                X = np.array([m + sigma * B @ (D * np.random.randn(d)) for _ in range(lam)])
                F = np.array([evaluate(x) for x in X])
                evals_in_cma += lam
                
                # Update local best
                idx_best = np.argmin(F)
                if F[idx_best] < local_best_f:
                    local_best_f = F[idx_best]
                    local_best_x = X[idx_best].copy()
                    last_improvement_eval = evals_in_cma
                
                # Sort
                idx = np.argsort(F)
                x_sorted = X[idx]
                
                # Weighted mean
                m_new = sum(w[i] * x_sorted[i] for i in range(mu))
                
                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))
                
                # Sigma update
                sigma *= np.exp(c_sigma * (np.linalg.norm(pc) / chi_d - 1))
                
                # Update C
                delta_h = 1.0 if np.linalg.norm(pc) / chi_d < (1.4 + 2/(d+1)) else 0.0
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C)
                for i in range(mu):
                    diff = x_sorted[i] - m_new
                    C += c_mu * w[i] * np.outer(diff, diff)
                
                # Symmetrize and decompose
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    B = np.eye(d)
                    D = np.ones(d)
                    C = np.eye(d)
                    
                m = m_new
                
                # Stagnation Check and Restart Strategy
                if evals_in_cma - last_improvement_eval >= stagnation_threshold:
                    # Restart: reset to local best, increase sigma
                    m = local_best_x.copy()
                    sigma = max(sigma * 1.5, 0.5)
                    pc = np.zeros(d)
                    C = np.eye(d)
                    B = np.eye(d)
                    D = np.ones(d)
                    last_improvement_eval = evals_in_cma
                    # If sigma is very large and no progress, shrink it?
                    # For Lunacek, we might be stuck in a local minimum.
                    # A bigger sigma helps escape.
                    
            return local_best_x, local_best_f

        # Strategy: Multi-start CMA-ES
        # 1. Initial exploration with a few random starts
        # 2. Refine the best start
        # 3. If stuck, try another start
        
        # Budget allocation:
        # Phase 1: 3 random starts, 50 evals each (150 total)
        # Phase 2: Take the best, run CMA-ES with remaining budget
        
        # However, Lunacek has two basins. Random starts might miss one.
        # Let's try a few diverse starts.
        
        # Start 1: Center
        m0 = np.zeros(d)
        f0 = evaluate(m0)
        evals_used += 1 # Wait, evaluate increments it.
        
        # Let's just run CMA-ES from center with a decent budget first.
        # If it gets stuck, restart from random points.
        
        # Actually, a simpler robust strategy for BBOB f24:
        # Run CMA-ES from 0. If it stagnates, restart from a random point far away.
        
        # Let's use the run_cma_es helper.
        # We need to be careful with evals_used tracking inside run_cma_es.
        # run_cma_es uses nonlocal evals_used.
        
        # Phase 1: Run CMA-ES from origin with 200 evals
        x1, f1 = run_cma_es(np.zeros(d), 2.0, 200, timeout - (time.time() - start_time))
        
        if time_left() <= 1.0:
            return best_x, best_f
            
        # Phase 2: If we didn't get close to 0 (or just always), try a second start
        # Lunacek's second basin is often around -1.1 * ones(d) or similar.
        # Let's try a few random starts from the remaining budget.
        
        remaining_budget = budget - evals_used
        
        # If we have enough budget, run another CMA-ES from a random point
        if remaining_budget > 100:
            # Random start
            m2 = np.random.uniform(-4, 4, d)
            x2, f2 = run_cma_es(m2, 2.0, remaining_budget, timeout - (time.time() - start_time))
            
        # Final polish: (1+1)-ES from the global best
        if time_left() > 1.0 and evals_used < budget:
            m = best_x.copy()
            sigma = 0.5
            for _ in range(50):
                if evals_used >= budget or time_left() < 0.5:
                    break
                y = m + sigma * np.random.randn(d)
                fy = evaluate(y)
                if fy < best_f:
                    m = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
