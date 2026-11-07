"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with Nelder-Mead refinement for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            # Ensure x is a valid 1D array of the correct dimension
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
            return fx

        # --- Phase 1: Multi-start CMA-ES ---
        
        # CMA-ES Parameters
        # For dim=5, a population size of 9-12 is typical. 
        # We use a smaller population to save budget for multiple starts or NM.
        # Let's use lambda=9, mu=4.
        lam = 9
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # CMA-ES state
        def run_cma_es(x0, sigma0, max_calls):
            nonlocal calls_used, best_x, best_f, deadline
            
            if time.time() > deadline or calls_used >= budget:
                return best_x, best_f
            
            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (20 * dim ** 2))
            
            # Initial population
            pop = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            # Generate initial population
            for i in range(lam):
                # Standard normal
                z = np.random.randn(dim)
                # Transform
                # Efficient transformation using Cholesky or eigendecomposition
                # For small dim, eigendecomposition is fine
                # C = V * diag(eigvals) * V.T
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-16)
                pop[i] = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                pop_f[i] = eval_point(pop[i])
                
                if time.time() > deadline or calls_used >= budget:
                    return best_x, best_f
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Main loop
            while calls_used < budget and time.time() < deadline:
                # Update mean
                x_old = x_mean
                x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # Update covariance
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T) + cmu * sum(weights * (yi * yi.T))
                # yi = (xi - x_old) / sigma
                y = (pop[:mu] - x_old) / sigma
                
                # Cumulation
                pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (x_mean - x_old) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C @ (x_mean - x_old) / sigma
                
                # Delta term for step size control
                delta_term = np.dot(ps, ps) / chiN ** 2 - (1 - (1 - cs) ** (1 / max(1, int((calls_used - budget/lam)))) )
                # Simplified hsig check
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 / max(1, int((calls_used - budget/lam))))) < (1.4 + 2 / (dim + 1)) * chiN
                if hsig:
                    pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (x_mean - x_old) / sigma
                
                # Update C
                c1 = 2 / ((dim + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum(weights[:, None, None] * y[:, :, None] * y[:, None, :], axis=0)
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
                # Check convergence
                if sigma < 1e-12 or np.linalg.norm(x_mean - x_old) < 1e-12:
                    break
                
                # Generate new population
                for i in range(lam):
                    z = np.random.randn(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-16)
                    pop[i] = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                    pop_f[i] = eval_point(pop[i])
                    
                    if time.time() > deadline or calls_used >= budget:
                        return best_x, best_f
                
                # Sort
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]
                
                # Optional: Early stop if no improvement for many generations
                # This is handled by the outer loop restarts or budget exhaustion
                
            return best_x, best_f

        # Nelder-Mead Refinement
        def run_nelder_mead(x0, max_calls):
            nonlocal calls_used, best_x, best_f, deadline
            
            if time.time() > deadline or calls_used >= budget:
                return best_x, best_f
            
            # Initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex_f = np.zeros(dim + 1)
            
            simplex[0] = x0
            simplex_f[0] = eval_point(x0)
            
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] = 0.5 # Initial step size
                simplex_f[i + 1] = eval_point(simplex[i + 1])
                
                if time.time() > deadline or calls_used >= budget:
                    return best_x, best_f
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls_used < budget and time.time() < deadline:
                # Sort
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]
                
                # Check convergence
                if np.std(simplex_f) < 1e-12:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = eval_point(x_ref)
                
                if time.time() > deadline or calls_used >= budget:
                    return best_x, best_f
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = eval_point(x_exp)
                    
                    if time.time() > deadline or calls_used >= budget:
                        return best_x, best_f
                        
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = eval_point(x_con)
                    
                    if time.time() > deadline or calls_used >= budget:
                        return best_x, best_f
                        
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_point(simplex[i])
                            
                            if time.time() > deadline or calls_used >= budget:
                                return best_x, best_f
            
            return best_x, best_f

        # Main Strategy:
        # 1. Use a significant portion of the budget for CMA-ES to find a good basin.
        # 2. Use the remainder for Nelder-Mead to refine.
        # 3. If CMA-ES stagnates, restart CMA-ES from a random point or the current best.
        
        # Let's allocate 60% of budget to CMA-ES and 40% to NM.
        # But we should be adaptive. 
        # Strategy: Run CMA-ES until it stagnates or uses 50% of budget.
        # Then run NM on the best point.
        # If NM finishes and we still have budget, maybe restart CMA-ES from best point with smaller sigma?
        # Or just run NM again?
        
        # Let's try:
        # 1. CMA-ES for up to 300 calls.
        # 2. Nelder-Mead for the remaining calls.
        
        cma_budget = int(budget * 0.6)
        nm_budget = budget - cma_budget
        
        # Run CMA-ES
        x_init = np.zeros(dim)
        # Maybe start CMA-ES at a random point to avoid bad local minima?
        # f13 is sharp ridge, usually centered or shifted.
        # Let's start at 0.
        run_cma_es(x_init, 1.0, cma_budget)
        
        # Run Nelder-Mead on the best point found
        if calls_used < budget and time.time() < deadline:
            run_nelder_mead(best_x, nm_budget)
            
        # If we have more time/budget, we could do another round of CMA-ES with smaller sigma
        # or another NM restart.
        # Given the 300s limit and 500 calls, we likely won't hit the time limit unless eval is slow.
        # Assuming eval is fast, we might have unused budget if NM converges quickly.
        # Let's check if we can do a second phase.
        
        if calls_used < budget and time.time() < deadline:
            # Second phase: CMA-ES with smaller sigma centered at best_x
            # Or just restart NM with a larger initial simplex?
            # Let's try a smaller CMA-ES refinement
            # Allocate remaining budget to CMA-ES with sigma = 0.1
            run_cma_es(best_x, 0.1, budget - calls_used)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
