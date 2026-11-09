import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES with proper implementation ---
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        gen = 0
        
        def cholesky_safe(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-10 * np.eye(n))
                return L
            except:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))
        
        cmaes_budget_limit = total_budget - 60
        no_improve_count = 0
        prev_best = np.inf
        
        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break
            
            L = cholesky_safe(C)
            z = np.random.standard_normal((lambda_, n))
            x_pop = mean[None, :] + sigma * (z @ L.T)
            
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])
            
            if calls_made >= cmaes_budget_limit:
                break
                
            idx = np.argsort(f_pop)
            x_w = sum(weights[i] * x_pop[idx[i]] for i in range(mu))
            
            # Check for convergence/stagnation
            if best_f < prev_best - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            prev_best = best_f
            
            if no_improve_count > 15 or sigma < 1e-6:
                break
            
            # Update p_sigma using C^{-1/2} @ (x_w - mean) / sigma
            # C^{-1/2} = L^{-1} since C = L @ L.T, so C^{-1/2} = L^{-1}
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except:
                C_inv_sqrt = np.eye(n)
                
            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2/(n+1)) * np.sqrt((mueff+1)/(mueff+3))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma
            
            # Update C
            C_update = sum(weights[i] * np.outer((x_pop[idx[i]] - mean)/sigma, (x_pop[idx[i]] - mean)/sigma) for i in range(mu))
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
            
            mean = x_w
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            gen += 1

        # --- Phase 2: Nelder-Mead + coordinate descent polish ---
        if best_x is not None and calls_made < total_budget:
            # Nelder-Mead simplex method
            x0 = best_x.copy()
            f0 = best_f
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
                f_simplex.append(evaluate(xi))
            
            # Nelder-Mead iterations
            max_nm_iter = 30
            for _ in range(max_nm_iter):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = evaluate(x_ref)
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = evaluate(x_con)
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = evaluate(simplex[i])
                
                if calls_made >= total_budget:
                    break
                
                # Check convergence
                if np.std(f_simplex) < 1e-8:
                    break

            # Final coordinate descent with adaptive step
            if calls_made < total_budget:
                step = 0.05
                for _ in range(50):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    improved = False
                    for dim_idx in range(n):
                        if calls_made >= total_budget:
                            break
                        
                        x_pos = best_x.copy()
                        x_pos[dim_idx] += step
                        x_pos = np.clip(x_pos, -5, 5)
                        f_pos = evaluate(x_pos)
                        if f_pos < best_f:
                            improved = True
                            continue
                        
                        x_neg = best_x.copy()
                        x_neg[dim_idx] -= step
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = evaluate(x_neg)
                        if f_neg < best_f:
                            improved = True
                    
                    if not improved:
                        step *= 0.5
                        if step < 1e-6:
                            break
                    else:
                        step = max(step * 0.8, 1e-6)

        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
