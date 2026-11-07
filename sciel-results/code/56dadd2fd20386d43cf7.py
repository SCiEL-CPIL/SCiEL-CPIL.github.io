"""Optimizer for BBOB f7 (Step ellipsoidal).
Hybrid: population-based adaptive-step global search (robust on flat plateaus)
drives exploration; Nelder-Mead simplex refines the best point when budget remains.
A strict wall-clock deadline guard returns the best result found safely early."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive search with Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        max_calls = self.budget  # 500

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        # ---------- Nelder-Mead local refinement ----------
        def nelder_mead(func, x0, deadline, rng):
            """Run NM simplex from x0. Updates best_x/best_f/calls_made in place."""
            nonlocal best_x, best_f, calls_made
            n = dim
            if not elapsed_ok():
                return
            alpha = 0.3
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += alpha
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if not elapsed_ok():
                    return
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            alpha_r, alpha_e, alpha_c, alpha_sh = 1.0, 2.0, 0.5, 0.5
            max_iter = 120
            tol = 1e-12
            it = 0
            while it < max_iter:
                if not elapsed_ok():
                    return
                it += 1
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                if fvals[-1] - fvals[0] < tol:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha_r * (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < fvals[-1]:
                    if fr < fvals[0]:
                        xe = np.clip(centroid + alpha_e * (centroid - simplex[-1]), -5, 5)
                        fe = func(xe)
                        calls_made += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    xc = np.clip(centroid + alpha_c * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + alpha_sh * (simplex[i] - simplex[0]), -5, 5)
                        for i in range(1, n + 1):
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if not elapsed_ok():
                            return
                mi = np.argmin(fvals)
                if fvals[mi] < best_f:
                    best_f = fvals[mi]
                    best_x = simplex[mi].copy()

        # ---------- Initial exploration: small diverse population ----------
        pop_size = 10
        candidates = []
        for _ in range(pop_size):
            if not elapsed_ok():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        candidates.sort(key=lambda t: t[0])
        x = candidates[0][1].copy()
        fx = candidates[0][0]

        # ---------- CMA-ES Main Loop ----------
        # Initialize CMA-ES parameters based on the best candidate found in initial exploration
        # We start CMA-ES from the best point found so far to exploit local structure
        # while maintaining global adaptation capabilities.
        
        # If we haven't found a good starting point, use center or random
        if best_x is None:
            x_mean = np.zeros(dim)
        else:
            x_mean = best_x.copy()
            
        # Initial sigma: heuristic based on dimension and box size
        sigma = 2.0 
        
        # CMA-ES parameters
        n = dim
        mu = n // 2 + 1  # number of parents
        # Weights: linear weights
        w = np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = w[:mu]
        mu_eff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + cs
        
        # State variables
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Precompute square root of C for sampling
        # We will update sqrt(C) periodically or use Cholesky
        # For simplicity and robustness, we will recompute Cholesky every iteration if budget allows, 
        # but that's expensive. Instead, we use a simplified update or just sample from standard normal 
        # and transform if we had a full CMA-ES. 
        # Given the budget is 500 and dim is 5, we can afford Cholesky.
        
        def chol_update(C, c1, cmu, w, y):
            # Rank-one and rank-mu updates
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum([wi * np.outer(yi, yi) for wi, yi in zip(w, y)], axis=0)
            # Ensure symmetry
            C = (C + C.T) / 2.0
            return C
        
        def sample_from_cma(mean, sigma, C, rng):
            # Standard normal
            z = rng.standard_normal(n)
            # Transform using Cholesky
            # We need sqrt(C). 
            # To avoid frequent Cholesky failures, we ensure C is positive definite.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add identity
                L = np.linalg.cholesky(C + 1e-6 * np.eye(n))
            x = mean + sigma * (L @ z)
            return np.clip(x, -5, 5)

        # Main CMA-ES loop
        # We run until budget is exhausted or convergence
        while elapsed_ok():
            # Sample lambda offspring
            lambda_pop = max(1, int(10 * (1.0 + mu_eff) / n)) # Heuristic population size
            if calls_made + lambda_pop > max_calls:
                lambda_pop = max_calls - calls_made
            if lambda_pop <= 0:
                break
                
            offspring = []
            for _ in range(lambda_pop):
                if not elapsed_ok():
                    break
                z = rng.standard_normal(n)
                # Update Cholesky only if necessary? For small dim, every time is fine.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(C + 1e-6 * np.eye(n))
                
                x_off = x_mean + sigma * (L @ z)
                x_off = np.clip(x_off, -5, 5)
                f_off = func(x_off)
                calls_made += 1
                offspring.append((f_off, x_off))
                
                if f_off < best_f:
                    best_f = f_off
                    best_x = x_off.copy()
            
            if not offspring:
                break
                
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Select mu best
            y_best = np.array([x for _, x in offspring[:mu]])
            
            # Update mean
            x_mean_new = np.sum([w[i] * y_best[i] for i in range(mu)], axis=0)
            
            # Update step size and path
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * C^{-1/2} * (x_mean_new - x_mean)/sigma
            # We need C^{-1/2}. Approximation: use inverse of Cholesky or just identity if C is close to I.
            # For robustness and speed, we can approximate C^{-1/2} as I if C is not well-conditioned, 
            # or compute it. Given dim=5, computation is cheap.
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L.T) # (L.T @ L)^-1/2 = L^-T
            except:
                C_inv_sqrt = np.eye(n)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (x_mean_new - x_mean) / sigma)
            
            if np.linalg.norm(ps) / chi_n < (1.4 + 1 / cs) * cc:
                pc = (1 - cc) * pc + cc * (x_mean_new - x_mean) / sigma
            else:
                pc = (1 - cc) * pc
                
            # Update Covariance
            C = chol_update(C, c1, cmu, w, y_best - x_mean)
            
            # Update Mean
            x_mean = x_mean_new
            
            # Update Sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Convergence check
            if sigma < 1e-12:
                break
                
            # Optional: Restart if stuck? 
            # If we are stuck for a long time, we might want to restart CMA-ES from best_x with larger sigma.
            # But standard CMA-ES handles this via adaptation.

        # ---------- Refine the best point with Nelder-Mead ----------
        if best_x is not None and elapsed_ok():
            nelder_mead(func, best_x.copy(), deadline, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
