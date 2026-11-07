"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # CMA-ES parameters
        dim = self.dim
        budget = self.budget
        
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lam // 2  # Selection size
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Initial parameters
        # Start near center, as f6 often has structure around origin or shifted
        # But hidden shift is in [-4, 4]. Uniform start is safer.
        rng = np.random.default_rng(42)
        
        # Try a few restarts if budget allows, or just one long run?
        # With 500 calls, a single CMA-ES run with lam=10 gives 50 generations.
        # This is usually sufficient for dim=5.
        
        # Initial mean
        x_mean = rng.uniform(-5, 5, dim)
        
        # Initial covariance: identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Initial step size
        sigma = 1.0
        
        # Accumulators for adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        def _update_covariance(C, C_inv, C_sqrt, C_inv_sqrt, weights, y_norm, p_sigma, p_c, sigma, c_sigma, c_c, c_1, c_mu, dim, mueff):
            # This is a simplified update. For robustness, we might just use 
            # a simpler adaptation or rely on the standard formulas.
            # However, implementing full CMA-ES correctly is tricky.
            # Let's use a simpler, robust alternative: 
            # **Nelder-Mead** is often better for <10 dims with small budgets 
            # if CMA-ES is not perfectly tuned.
            
            # Actually, let's stick to a well-tuned Nelder-Mead with restarts.
            # It is simpler to implement correctly and often outperforms 
            # poorly tuned CMA-ES on small budgets.
            
            pass
        
        # Let's pivot to Multi-Start Nelder-Mead.
        # It is robust and handles the "attractive sector" well if the 
        # simplex is large enough.
        
        def nelder_mead(func, x0, max_iter, timeout_check=None):
            # Initialize simplex
            dim = len(x0)
            simplex = [x0.copy() for _ in range(dim + 1)]
            for i in range(dim):
                simplex[i+1][i] += 0.25 * (abs(x0[i]) + 1e-1)
            
            fvals = [func(x) for x in simplex]
            best_x = simplex[0]
            best_f = min(fvals)
            
            # Standard Nelder-Mead loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            for _ in range(max_iter):
                if timeout_check and timeout_check():
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                best_x = simplex[0]
                best_f = fvals[0]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fc = func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = func(xc)
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                
                # Clip to bounds
                for i in range(len(simplex)):
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    # If clipped, re-evaluate? No, func clips internally.
                    # But we need to update fvals if we clipped?
                    # The problem states func(x) clips. So we don't need to 
                    # re-evaluate if we just clip the stored x.
                    # However, Nelder-Mead relies on the geometry. 
                    # Clipping can distort the simplex.
                    # It's safer to let func handle it.
            
            return best_x, best_f

        # Multi-Start Strategy
        # Use 10% of budget for initial exploration, 90% for refinement?
        # Or just split budget among 5 restarts?
        # f6 is multimodal-ish. 5 restarts with 100 evals each seems good.
        
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        best_x_global = None
        best_f_global = np.inf
        
        for r in range(num_restarts):
            # Random start
            x_start = rng.uniform(-5, 5, dim)
            
            # Run Nelder-Mead
            x_res, f_res = nelder_mead(func, x_start, budget_per_restart, 
                                       timeout_check=lambda: time.time() > deadline)
            
            if f_res < best_f_global:
                best_f_global = f_res
                best_x_global = x_res
        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
