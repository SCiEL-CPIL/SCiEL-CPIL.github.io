"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES for global search and Nelder-Mead for local polish.
    Designed for BBOB f14 in 5 dimensions with a 500-call budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 325s to ensure we return.
        deadline = t_start + 325.0 
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # State variables
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return (deadline - time.time()) > 2.0  # 2s buffer for final steps/return

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # For dim=5, standard recommended lambda is ~ 4 + 3*ln(5) ~ 8.8 -> 9
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        
        # Weights for the best mu individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cs = (mueff + 1.5) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) / (1 - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        # Start centered, with a moderate sigma to cover [-5, 5] reasonably
        mean = np.zeros(dim)
        sigma = 2.0 
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-calculate Cholesky factor if possible, or update in loop
        L = np.eye(dim)
        
        while calls < budget and time_left():
            # 1. Generate Population
            # We generate all candidates at once for efficiency
            Z = rng.standard_normal((lam, dim))
            
            # Update Cholesky factor of C if it changed significantly or initially
            # In a simplified CMA-ES, we often recompute L each step or use a stable factorization.
            # Given the small dim, recomputing is fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add jitter
                C = (C + C.T) / 2.0 + 1e-10 * np.eye(dim)
                L = np.eye(dim)
            
            # Y = Mean + Sigma * Z @ L.T  (Standard CMA-ES update)
            Population = mean + sigma * (Z @ L.T)
            
            # Clip to box
            Population = np.clip(Population, -5, 5)
            
            # Check if we can evaluate the whole population
            if calls + lam > budget:
                # If not enough budget for full population, evaluate only as many as we can
                # This effectively acts as a (1+1)-ES or smaller population ES for the last few steps
                remaining = budget - calls
                if remaining <= 0:
                    break
                Pop = Population[:remaining]
                calls_needed = remaining
            else:
                Pop = Population
                calls_needed = lam
            
            # Evaluate Population
            Fvals = []
            for i in range(calls_needed):
                x = Pop[i]
                f = func(x)
                calls += 1
                Fvals.append(f)
                update_best(x, f)
                
                # Early exit if time runs out mid-evaluation
                if not time_left():
                    break
            
            if calls >= budget or not time_left():
                break
            
            # If we didn't evaluate the full population, we can't do a full CMA-ES update
            # We just update the mean to the best point found in this truncated batch
            if calls_needed < lam:
                # (1+1)-ES style update for the last few calls
                # Find best in Fvals
                best_idx = np.argmin(Fvals)
                best_val = Fvals[best_idx]
                best_x_pop = Pop[best_idx]
                
                if best_val < best_f:
                    mean = best_x_pop.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                    # Keep mean as is, or move slightly towards best?
                    # Standard (1+1) keeps mean if step fails.
                continue
            
            # 2. Sort and Update CMA-ES Parameters
            # Indices of the best mu individuals
            order = np.argsort(Fvals)
            best_indices = order[:mu]
            
            # New Mean
            new_mean = np.average(Pop[best_indices], axis=0, weights=weights)
            
            # Step size control (PS)
            # y_old = (new_mean - mean) / (sigma * norm(L))
            norm_L = np.linalg.norm(L, 'fro')
            if norm_L > 1e-16:
                y_old = (new_mean - mean) / (sigma * norm_L)
            else:
                y_old = np.zeros(dim)
            
            # Update PS
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_old
            
            # HSig condition
            # Approximation for E[||N(0, I)||]
            E_norm = np.sqrt(dim) * (1 - 1.0/(4*dim) + 3.0/(32*dim*dim))
            # The denominator term (1 - (1-cs)^(t / (cs*(d+1)))) is an approximation for the running norm
            # For simplicity and robustness, we use a simpler heuristic or the standard one if t is available.
            # Let's use the standard formula with current iteration count approximation.
            # Iterations approx = calls / lam
            iter_count = calls / lam
            denom = 1 - (1 - cs) ** (iter_count / (cs * (dim + 1)))
            if denom < 1e-16:
                denom = 1e-16
            hsig = (np.linalg.norm(ps) / np.sqrt(denom)) < (1.4 + 2 / (dim + 1)) * E_norm
            
            # Update PC
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_old
            
            # Update Covariance C
            # Rank-mu update
            Z_best = (Pop[best_indices] - mean) / (sigma * norm_L)
            # Weighted outer product
            # rank_mu = sum(weights[i] * outer(Z_best[i], Z_best[i]))
            rank_mu = np.sum([weights[i] * np.outer(Z_best[i], Z_best[i]) for i in range(mu)], axis=0)
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update Sigma
            norm_ps = np.linalg.norm(ps)
            sigma = sigma * np.exp(cs * (norm_ps / E_norm - 1))
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-12, 50.0)
            
            mean = new_mean
            
            # Convergence check
            if sigma < 1e-12:
                break

        # 3. Nelder-Mead Polish Phase
        # Use remaining budget to refine the best solution found by CMA-ES
        if calls < budget and time_left():
            nm_x = best_x.copy()
            nm_f = best_f
            
            # Initial Simplex
            # Start with best point, and perturb each dimension
            step = 0.5
            simplex = [nm_x.copy()]
            simplex_f = [nm_f]
            
            for i in range(dim):
                x_i = nm_x.copy()
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                
                # Evaluate
                if calls < budget and time_left():
                    f_i = func(x_i)
                    calls += 1
                    simplex_f.append(f_i)
                    update_best(x_i, f_i)
                else:
                    # If we run out of budget before completing simplex, break
                    # Note: NM requires dim+1 points. If we don't have them, we skip NM.
                    break
            
            # If we have a complete simplex, run NM
            if len(simplex) == dim + 1:
                # Nelder-Mead Loop
                max_nm_iter = 50  # Limit iterations to save time for safety
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                for _ in range(max_nm_iter):
                    if calls >= budget or not time_left():
                        break
                    
                    # Sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    x_worst = simplex[-1]
                    f_worst = simplex_f[-1]
                    x_best = simplex[0]
                    f_best = simplex_f[0]
                    x_second = simplex[1]
                    f_second = simplex_f[1]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    
                    if calls < budget and time_left():
                        f_r = func(x_r)
                        calls += 1
                        update_best(x_r, f_r)
                    else:
                        break
                    
                    if f_r < f_best:
                        # Expansion
                        x_e = centroid + gamma * (centroid - x_worst)
                        x_e = np.clip(x_e, -5, 5)
                        
                        if calls < budget and time_left():
                            f_e = func(x_e)
                            calls += 1
                            update_best(x_e, f_e)
                            
                            if f_e < f_r:
                                simplex[-1] = x_e
                                simplex_f[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                simplex_f[-1] = f_r
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                            break
                    elif f_r < f_second:
                        # Accept Reflection
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contraction
                        x_c = centroid + rho * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        
                        if calls < budget and time_left():
                            f_c = func(x_c)
                            calls += 1
                            update_best(x_c, f_c)
                            
                            if f_c < f_worst:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = x_best + sigma_shrink * (simplex[i] - x_best)
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    
                                    if calls < budget and time_left():
                                        f_i = func(simplex[i])
                                        calls += 1
                                        simplex_f[i] = f_i
                                        update_best(simplex[i], f_i)
                                    else:
                                        break
                                # Re-sort after shrink
                                order = np.argsort(simplex_f)
                                simplex = [simplex[i] for i in order]
                                simplex_f = [simplex_f[i] for i in order]
                                break # Usually break after shrink to avoid wasting budget
                        else:
                            break
                    
                    # Convergence Check
                    if max(simplex_f) - min(simplex_f) < 1e-10:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
