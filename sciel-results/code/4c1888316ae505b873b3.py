"""CMA-ES Optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with restarts and local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        evals_used = 0
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8. 
        # Let's use a smaller lambda for efficiency with 500 budget, maybe 5 or 6.
        # Standard recommendation: lambda >= 2*n. For n=5, lambda=10 is good, but 500/10=50 generations.
        # Let's try lambda=6.
        lam = 6
        
        # Reserve budget for polishing
        polish_budget = min(50, budget // 10)
        search_budget = budget - polish_budget

        def cma_es_run(x_start, sigma_init, max_evals):
            nonlocal evals_used, best_f, best_x
            nonlocal deadline
            
            if evals_used >= max_evals:
                return
            
            # Initialize CMA-ES state
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            
            # Update rates
            mu = lam // 2  # Weight top mu individuals
            # Weights: linear or proportional? Let's use simple proportional weights for top mu
            # Actually, standard CMA-ES uses specific weights. 
            # w_i = log(mu+0.5) - log(i) for i=1..mu, normalized.
            weights = np.zeros(lam)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Adaptation rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # State variables
            pc = np.zeros(n)
            ps = np.zeros(n)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            
            # Initial evaluation
            if evals_used < max_evals and time.time() < deadline:
                fx = func(x_mean)
                evals_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_mean.copy()
            else:
                return

            # Main loop
            while evals_used < max_evals:
                if time.time() > deadline:
                    break
                
                # Sample population
                y = np.zeros((lam, n))
                for i in range(lam):
                    # Cholesky factorization of C (approximate update)
                    # For small n, we can just do Cholesky each time or use a cheaper update.
                    # Since n=5, Cholesky is cheap.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset
                        C = np.eye(n)
                        L = np.eye(n)
                        break
                    
                    z = rng.standard_normal(n)
                    y[i] = x_mean + sigma * (L @ z)
                
                # Evaluate population
                fvals = np.zeros(lam)
                for i in range(lam):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    fvals[i] = func(y[i])
                    evals_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = y[i].copy()
                
                if evals_used >= max_evals or time.time() > deadline:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                y_sorted = y[idx]
                fvals_sorted = fvals[idx]
                
                # Update mean
                x_mean_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # Check if we should update C (usually every step)
                # Update ps and pc
                # Need C^-1/2 @ (x_mean_new - x_mean)
                # We maintain C_inv_sqrt. 
                # Simple approach for small n: compute inverse sqrt via eigendecomp or Cholesky inverse.
                # C = L @ L.T. C^-1/2 = (L.T)^-1 @ L^-1? No.
                # Let's just compute the inverse square root of C using eigendecomposition for simplicity and robustness, 
                # as n=5 is very small.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * (1 - cs)
                
                # Update pc
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                
                # Update C
                delta_d = hsig * (2 - cc) - 1
                C = (1 + c1 * delta_d - c1 - cmu) * C + \
                    c1 * np.outer(pc, pc) + \
                    cmu * np.sum(weights[:mu][:, None, None] * np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean) / (sigma**2), axis=0)
                
                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                
                # Check for stagnation or divergence
                if sigma > 1e6 or sigma < 1e-12:
                    break
                if np.linalg.norm(x_mean_new - x_mean) < 1e-10 * sigma:
                    # Stagnation in mean
                    # If we are stuck, break to restart
                    break

                x_mean = x_mean_new

        # Multi-start CMA-ES
        # We will run several short CMA-ES instances from different starting points
        # or one long instance with restarts.
        # Given the multi-modality, multi-start is better.
        # Let's divide the search budget into chunks for different starts.
        # 5 starts, 80 evals each? Or adaptive?
        # Let's do 3 starts.
        num_starts = 3
        evals_per_start = search_budget // num_starts
        
        for i in range(num_starts):
            if evals_used >= search_budget:
                break
            
            # Start point: Random or based on previous best?
            # Random is better for exploration in multi-modal.
            # But we can bias towards the current best if we are doing a refinement.
            # For the first start, use center. For others, random.
            if i == 0:
                x_start = np.zeros(n)
            else:
                x_start = rng.uniform(-5, 5, n)
            
            # Initial sigma: 1.0 is reasonable for [-5,5]
            sigma_init = 1.0
            
            # Run CMA-ES for this chunk
            cma_es_run(x_start, sigma_init, search_budget)
            
            # If we found a good point, we might want to restart from it with smaller sigma?
            # The current cma_es_run handles adaptation. 
            # If it stagnated, we just move to the next start.
            
        # Polish Phase
        remaining = self.budget - evals_used
        if remaining > 10 and best_x is not None and time.time() < deadline:
            # Use Nelder-Mead or a simple local search
            # Nelder-Mead is often effective for final polishing in BBOB.
            from scipy.optimize import minimize
            
            nm_evals = 0
            def nm_func(x_nm):
                nonlocal nm_evals
                if nm_evals >= remaining:
                    return best_f
                x_clipped = np.clip(x_nm, -5, 5)
                val = func(x_clipped)
                nm_evals += 1
                return val

            try:
                # Initialize simplex around best_x
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxfev': remaining, 'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
