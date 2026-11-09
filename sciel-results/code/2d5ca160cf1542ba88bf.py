import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        eval_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global exploration
        # Start with a reasonable initial point and use CMA-ES
        x0 = np.zeros(d)
        f0 = evaluate(x0)
        
        # CMA-ES parameters
        sigma = 2.0
        mu = max(1, d // 2)  # number of parents
        lambda_pop = 2 * mu  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        pc = np.zeros(d)
        ps = np.zeros(d)
        hs = (1.0 + 1.5 + np.sqrt((mu_eff - 1) / (d + 1))) / (d + 1)
        
        # Initial covariance
        C = np.eye(d)
        # Precompute eigen decomposition
        eigval, eigvec = np.linalg.eigh(C)
        D = 1.0 / np.sqrt(eigval)
        B = eigvec
        Cinv_sqrt = (B * D) @ B.T
        
        # Phase 1: Run CMA-ES for ~60% of budget
        phase1_budget = int(budget * 0.6)
        max_iter = (phase1_budget - 1) // lambda_pop
        
        for gen in range(max_iter):
            if time.time() > deadline or eval_count >= budget - 2 * lambda_pop:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_pop):
                # Sample from multivariate normal
                z = rng.standard_normal(d)
                # Transform using B, D, and sigma
                y = B @ (D * z)
                x_new = x0 + sigma * y
                offspring.append(x_new)
            
            # Evaluate offspring
            offspring_f = [evaluate(x) for x in offspring]
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            selected = [offspring[i] for i in indices[:mu]]
            selected_f = [offspring_f[i] for i in indices[:mu]]
            
            # Weighted mean of selected
            x_new = np.sum([w * x for w, x in zip(weights, selected)], axis=0)
            
            # Update path covariance
            y_c = Cinv_sqrt @ (x_new - x0) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_c
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) <
                    (1 + 2 / (d + 1)) * hs)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum([
                (weights[i] / c_1) * np.outer(selected[i] - x0, selected[i] - x0) / sigma**2
                for i in range(mu)
            ], axis=0)
            
            C = (1 + c_1 * (c_mu - 1) + c_mu * delta_hsig) * C \
                - c_1 * rank_one + c_mu * rank_mu
            
            # Re-orthonormalize and update B, D
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-20, None)
            D = 1.0 / np.sqrt(eigval)
            B = eigvec
            Cinv_sqrt = (B * D) @ B.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / hs - 1))
            
            # Move to new mean
            x0 = x_new
            
            # Check convergence
            if np.max(eigval) / np.min(eigval) > 1e14:
                break
            if sigma < 1e-10:
                break
        
        # Phase 2: Local refinement with Nelder-Mead
        center = best_x.copy()
        simplex = [(center.copy(), best_f)]
        
        # Add vertices with adaptive step size
        step = 0.1
        for i in range(d):
            if time.time() > deadline or eval_count >= budget:
                break
            pt = center.copy()
            pt[i] += step
            fval = evaluate(pt)
            simplex.append((pt.copy(), fval))
        
        # Sort simplex
        simplex.sort(key=lambda p: p[1])
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        # Run NM for remaining budget
        nm_iterations = 0
        max_nm_iter = 300
        
        while nm_iterations < max_nm_iter:
            if time.time() > deadline or eval_count >= budget:
                break
            
            nm_iterations += 1
            
            # Get best, worst, second worst
            x_worst, f_worst = simplex[-1]
            x_second, f_second = simplex[-2]
            
            # Centroid of all but worst
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)
            
            # Reflection
            if eval_count + 1 > budget:
                break
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = evaluate(x_ref)
            
            if f_ref < simplex[0][1]:  # Better than best
                # Expansion
                if eval_count + 1 > budget:
                    break
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = evaluate(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < f_second:  # Between best and second worst
                simplex[-1] = (x_ref, f_ref)
            else:  # Worse than second worst
                # Contraction
                if eval_count + 1 > budget:
                    break
                if f_ref < f_worst:
                    # Outer contraction
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                else:
                    # Inner contraction
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                
                f_con = evaluate(x_con)
                
                if f_con < min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    if eval_count + d > budget:
                        break
                    
                    x_best, f_best_cur = simplex[0]
                    new_simplex = [(x_best, f_best_cur)]
                    shrink_completed = True
                    for i, (x_i, f_i) in enumerate(simplex):
                        if i == 0:
                            continue
                        if eval_count >= budget:
                            shrink_completed = False
                            break
                        x_shrink = x_best + rho * (x_i - x_best)
                        f_shrink = evaluate(x_shrink)
                        new_simplex.append((x_shrink, f_shrink))
                    
                    if shrink_completed:
                        simplex = new_simplex
                    else:
                        break
            
            # Re-sort
            simplex.sort(key=lambda p: p[1])
            
            # Convergence check
            x_min = np.min([p[0] for p in simplex], axis=0)
            x_max = np.max([p[0] for p in simplex], axis=0)
            if np.max(x_max - x_min) < 1e-10:
                break
        
        return best_x, best_f
