import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def check_deadline():
            return time.time() > deadline or calls_used >= total_budget

        def nelder_mead(x0, max_evals, rng):
            """Run Nelder-Mead optimization."""
            nonlocal calls_used, best_f, best_x
            
            if max_evals < dim + 2:
                return x0, best_f

            # Initialize simplex
            alpha = 1.0  # Expansion
            gamma = 2.0  # Reflection
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            # Initial simplex: x0 and x0 + delta * e_i
            delta = 0.5
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            f_vals = np.zeros(dim + 1)
            f_vals[0] = func(x0)
            calls_used += 1
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = x0.copy()
            
            for i in range(dim):
                simplex[i+1] = x0
                simplex[i+1][i] += delta
                f_vals[i+1] = func(simplex[i+1])
                calls_used += 1
                if f_vals[i+1] < best_f:
                    best_f = f_vals[i+1]
                    best_x = simplex[i+1].copy()

            evals = dim + 1
            max_iters = max_evals - evals
            
            for _ in range(max_iters):
                if check_deadline() or calls_used + dim + 1 > total_budget:
                    break
                    
                # Sort simplex
                idx = np.argsort(f_vals)
                simplex = simplex[idx]
                f_vals = f_vals[idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_vals[0] <= f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                elif f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + alpha * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
                            calls_used += 1
                            if f_vals[i] < best_f:
                                best_f = f_vals[i]
                                best_x = simplex[i].copy()
                        
                        if calls_used >= total_budget:
                            break

            return best_x, best_f

        def cma_es(x0, sigma0, max_evals, rng):
            """Run CMA-ES."""
            nonlocal calls_used, best_f, best_x
            
            if max_evals < 10:
                return x0, best_f, False # False indicates no convergence for polish

            lam = 10
            mu = lam // 2
            
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            L = np.linalg.cholesky(C)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_x = x_mean.copy()
            local_best_f = func(x_mean)
            calls_used += 1
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            evals_in_run = 1
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
            
            def update_cholesky(C_new):
                for eps in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
                    try:
                        return np.linalg.cholesky(C_new + eps * np.eye(dim))
                    except:
                        continue
                return np.eye(dim)
            
            stagnation = 0
            stagnation_threshold = 15
            
            converged = False
            
            while evals_in_run + lam <= max_evals:
                if calls_used + lam > total_budget:
                    break
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal((lam, dim))
                y = z @ L.T
                x_pop = x_mean + sigma * y
                x_pop = np.clip(x_pop, -5, 5)
                
                f_pop = np.array([func(x) for x in x_pop])
                evals_in_run += lam
                calls_used += lam
                
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < local_best_f:
                    local_best_f = f_pop[idx_best]
                    local_best_x = x_pop[idx_best].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                    stagnation = 0
                else:
                    stagnation += 1
                
                # Stagnation: Jump but keep C
                if stagnation > stagnation_threshold:
                    x_mean = local_best_x.copy()
                    sigma = max(sigma, 0.5) * 2.0
                    # Do NOT reset C, keep the rotation info
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    stagnation = 0
                    continue
                
                idx_sort = np.argsort(f_pop)
                x_sorted = x_pop[idx_sort]
                y_sorted = y[idx_sort]
                
                x_w = x_sorted[:mu]
                y_w = y_sorted[:mu]
                
                x_mean_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)
                
                d_mean = x_mean - x_mean_old
                
                try:
                    L_inv_T = np.linalg.inv(L).T
                except:
                    L_inv_T = np.eye(dim)
                    
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))
                
                gen = max(1, evals_in_run / lam)
                c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)
                
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 / gen))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
                
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))
                
                y_mean = np.sum(weights[:, np.newaxis] * y_w, axis=0)
                y_centered = y_w - y_mean[np.newaxis, :]
                
                C = (1 - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                    + c_mu * np.sum(weights[:, np.newaxis]**2 * y_centered[:, np.newaxis] * y_centered, axis=0)
                
                C = (C + C.T) / 2
                L = update_cholesky(C)
                
                if sigma < 1e-6:
                    converged = True
                    break
                    
            return local_best_x, local_best_f, converged
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling
        n_global = 10
        candidates = []
        
        for i in range(n_global):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1]
        
        # Phase 2: Main CMA-ES run
        remaining = total_budget - calls_used
        x_cma, f_cma, converged = cma_es(x0, 1.5, remaining, rng)
        
        # Phase 3: Polish with Nelder-Mead if CMA-ES converged or if we have budget left
        # We always try to polish if we have at least 10 evals left and CMA-ES finished
        if calls_used < total_budget:
            x_start = best_x if best_x is not None else x_cma
            remaining_for_nm = total_budget - calls_used
            # Reserve a few calls for safety
            if remaining_for_nm > 15:
                nelder_mead(x_start, remaining_for_nm, rng)

        return best_x, best_f
