import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin for 330s limit
        
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Track best
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Dense initial sampling to get a good start
        # Use a mix of uniform and Latin Hypercube-like sampling
        n_init = min(50, budget // 10)
        # Latin Hypercube Sampling
        perm = np.arange(n_init)
        lhs_points = np.zeros((n_init, n))
        for d in range(n):
            rng.shuffle(perm)
            lhs_points[:, d] = -5 + 10 * (perm + 0.5) / n_init
        
        for x in lhs_points:
            if calls_made >= budget:
                break
            eval_point(x)
        
        if calls_made >= budget:
            return best_x, best_f
        
        # Phase 2: CMA-ES
        # Implement a lightweight CMA-ES
        # Parameters
        pop_size = 10  # lambda
        sigma = 2.0    # Initial step size
        x_mean = best_x.copy()
        
        # CMA-ES weights
        mu = pop_size // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial matrices
        C = np.eye(n)
        C_inv = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        sqrt_eigvals = np.ones(n)
        
        # Precompute cholesky of C
        try:
            L = np.linalg.cholesky(C)
        except:
            L = np.eye(n)
        
        max_generations = (budget - calls_made) // pop_size
        if max_generations < 1:
            return best_x, best_f
        
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                # Apply transformation: x = x_mean + sigma * B * z
                # B is the square root of C
                x_new = x_mean + sigma * (L @ z)
                offspring.append((x_new, np.inf))
            
            # Evaluate offspring
            for i, (x_new, _) in enumerate(offspring):
                if calls_made >= budget:
                    break
                f = eval_point(x_new)
                offspring[i] = (x_new, f)
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * offspring[i][0]
            
            # Update path covariance
            sigma_old = sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * sigma / np.linalg.norm(C) * (new_mean - x_mean)
            
            # Update path sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L.T @ (new_mean - x_mean) / sigma
            
            # Check for sigma update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 1 / n) * np.sqrt(n - 1):
                sigma = sigma_old * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(n - 1) - 1))
                sigma = np.clip(sigma, 1e-10, 1e5)
            
            # Update covariance matrix
            # Delta h
            delta_h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 1 / n) * np.sqrt(n - 1) else 0
            
            rank_one = np.outer(new_mean - x_mean, new_mean - x_mean)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (offspring[i][0] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * delta_h * (n + 2 - mu_eff) - c_1 - c_mu) * C + c_1 * (rank_one / sigma**2 + (delta_h - 1) * pc @ pc / sigma**2) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update inverse
            try:
                C_inv = np.linalg.inv(C)
            except:
                C_inv = np.eye(n)
            
            # Update cholesky
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(n)
            
            # Update mean
            x_mean = new_mean
            
            # Check for convergence
            if sigma < 1e-6 or np.max(np.abs(C)) < 1e-10:
                break
        
        # Phase 3: Local refinement with Nelder-Mead if budget remains
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline - 5:
            # Use a simple local search around best_x
            # Nelder-Mead
            simplex = [best_x.copy()]
            for d in range(n):
                point = best_x.copy()
                step = 0.1 if abs(point[d]) > 1e-3 else 0.1
                point[d] += step
                simplex.append(point)
            
            simplex_f = [eval_point(simplex[0])]
            for i in range(1, n + 1):
                if calls_made >= budget:
                    break
                simplex_f.append(eval_point(simplex[i]))
            
            # Nelder-Mead iterations
            max_iter = min(remaining // (n + 1), 50)
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_point(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_point(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = eval_point(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= budget:
                                break
                            simplex_f[i] = eval_point(simplex[i])
        
        return best_x, best_f
