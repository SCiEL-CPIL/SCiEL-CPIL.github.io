"""CMA-ES with Restart Strategy for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with restarts.
    
    Strategy:
    1. Use CMA-ES because it adapts to rotated and ill-conditioned landscapes,
       which are characteristic of BBOB functions like Lunacek.
    2. Use a restart strategy (CMA-ES-RS) to handle multimodality. If the
       current run stagnates (stagnation detection), we restart with a new
       population centered on the best point found so far, with a slightly
       reduced step size or re-centered to explore new basins.
    3. Budget management: Allocate budget to runs, ensuring we stay within
       500 calls and 300s time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        max_time = 285.0  # Safety margin

        # Global best
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= total_budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            f_val = func(x)
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x.copy()
            return f_val

        # CMA-ES Parameters
        # Standard CMA-ES parameters for dim=5
        # lambda (population size) = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        if lam < 6: lam = 6
        
        # Initial step size
        sigma0 = 2.0  # Start wide to explore [-5, 5]^5
        
        # Restart logic parameters
        max_runs = 10 # Max number of CMA-ES runs
        min_calls_per_run = 50 # Minimum calls to consider a run valid before restarting

        best_x_so_far = np.zeros(dim)
        best_f_so_far = np.inf

        for run_idx in range(max_runs):
            if calls_used >= total_budget:
                break
            if time.time() - start_time > max_time:
                break

            # Determine starting point
            if run_idx == 0:
                # Start at center or random? Center is often good for BBOB if not shifted too far.
                # But Lunacek has a distant optimum. Let's start random-ish but bounded.
                # Actually, for the first run, a random start is safer to avoid getting stuck in a bad local min immediately.
                # However, the contract says minimizer is in [-4, 4].
                # Let's use a random start for the first run to ensure exploration.
                # Or better, use the global best from previous runs if any.
                # For run 0, no global best yet.
                x_mean = np.random.uniform(-3, 3, size=dim)
            else:
                # Restart around the best point found so far
                # Shrink the search region slightly or keep it?
                # Standard CMA-ES-RS restarts with a new random population, 
                # but often centered near the previous best or just random.
                # Given the "Lunacek" nature (two peaks), restarting randomly 
                # might find the other peak.
                # Let's restart with a new random center, but maybe biased towards 
                # the previous best? No, pure random restart is robust for multimodal.
                # However, to be efficient, let's center the new run's mean at the 
                # global best, but with a large sigma to allow escape.
                x_mean = global_best_x.copy()
                # Reduce sigma for restart to refine? Or increase to explore?
                # CMA-ES-RS typically resets the covariance and sigma.
                # Let's reset sigma to a medium value.
                sigma0 = 1.0 if run_idx > 1 else 2.0

            # Initialize CMA-ES state
            x = x_mean.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Update rates
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs

            # Initial population
            # Generate lam samples
            # We need to track the history for stagnation detection
            best_f_in_run = np.inf
            best_x_in_run = None
            calls_in_run = 0
            stagnation_count = 0
            threshold = 1e-10

            # Pre-allocate arrays for efficiency
            # We will generate samples one by one or in batches?
            # Batching is faster in NumPy.
            # But we need to check budget and time frequently.
            # Let's do batches of size lam.

            while calls_used < total_budget and (time.time() - start_time) < max_time:
                # Check if we should stop this run
                if calls_in_run > min_calls_per_run:
                    # Stagnation check: if best_f hasn't improved significantly in last few generations
                    # We track best_f_in_run. If it hasn't changed by threshold, increment stagnation.
                    # Simple heuristic: if best_f_in_run is very close to previous best, stop.
                    # Actually, CMA-ES has built-in stopping criteria (condition number, sigma).
                    # Let's implement a simple stagnation: if no improvement in last 20 * dim steps?
                    # To keep it simple and fast, let's just rely on the main loop condition 
                    # and restart if we detect convergence (small spread).
                    
                    # Check convergence: spread of population
                    # We don't have the current population stored easily in this loop structure without storing it.
                    # Let's store the last population.
                    pass

                # Generate population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C if it becomes singular
                    C = np.eye(dim)
                    L = np.eye(dim)

                # Generate standard normal samples
                # Shape: (lam, dim)
                zs = np.random.normal(0, 1, (lam, dim))
                
                # Transform to x-space
                # x_i = x_mean + sigma * zs @ L
                pop = x + sigma * (zs @ L)
                
                # Evaluate population
                # We must respect the budget.
                # If we don't have enough budget for the whole population, we stop.
                if calls_used + lam > total_budget:
                    # Evaluate only as many as we can afford
                    # But CMA-ES update needs full population.
                    # If we can't afford the full pop, we should stop the run.
                    break
                
                fs = []
                for i in range(lam):
                    f_val = safe_call(pop[i])
                    fs.append(f_val)
                    calls_in_run += 1
                    
                    if f_val < best_f_in_run:
                        best_f_in_run = f_val
                        best_x_in_run = pop[i].copy()
                    
                    # Early exit if time is tight?
                    if time.time() - start_time > max_time - 5:
                        break
                
                if len(fs) < lam:
                    break

                fs = np.array(fs)
                
                # Sort population by fitness
                idx = np.argsort(fs)
                pop = pop[idx]
                fs = fs[idx]
                
                # Update best
                if fs[0] < best_f_in_run:
                    best_f_in_run = fs[0]
                    best_x_in_run = pop[0].copy()

                # Update mean
                # x = sum(weights * pop[i] for i in range(mu))
                x_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update step size (sigma)
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) / sigma * L^-1 * (x_new - x)
                # L^-1 is inv(L)
                try:
                    inv_L = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    inv_L = np.eye(dim)
                    
                diff = x_new - x
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (inv_L @ diff)
                
                # hsig check
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_in_run + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 + cc)
                
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                # Update covariance matrix
                # pc = (1 - cc) * pc + hsig * cc * (x_new - x)
                pc = (1 - cc) * pc + hsig * cc * diff
                
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(weights * (diff_i * diff_i.T))
                # Note: The term (1 - hsig) * cc * (2 - cc) * C is an approximation for the rank-1 update from pc
                # Actually, the standard formula is:
                # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(w_i * (x_i - x) @ (x_i - x).T)
                
                # Rank-1 update from pc
                rank1 = pc @ pc.T
                # Correction term if hsig is False
                if not hsig:
                    rank1 += cc * (2 - cc) * C
                
                # Rank-mu update
                # diff_pop = pop[:mu] - x (old x)
                diff_pop = pop[:mu] - x
                # Weighted outer products
                # weights[:, np.newaxis, np.newaxis] * diff_pop[:, np.newaxis, :] * diff_pop[:, np.newaxis, :]
                rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * diff_pop[:, np.newaxis, :] * diff_pop[:, np.newaxis, :], axis=0)
                
                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update x
                x = x_new

                # Stopping criteria for this run
                # 1. Condition number of C is large
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    cond = eigvals[-1] / (eigvals[0] + 1e-16)
                    if cond > 1e14:
                        break
                except:
                    break
                
                # 2. Sigma is too small
                if sigma < 1e-12:
                    break
                    
                # 3. Spread of population is small
                # Approximate spread by std of the best mu points
                if np.std(pop[:mu], axis=0).max() < 1e-10:
                    break

            # End of run
            # Update global best if this run found something better
            if best_f_in_run < global_best_f:
                global_best_f = best_f_in_run
                global_best_x = best_x_in_run.copy()
                
            # If we broke out due to stagnation/convergence, we restart.
            # If we broke out due to budget/time, the outer loop will handle it.

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
