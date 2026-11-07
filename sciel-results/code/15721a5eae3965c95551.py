import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        def time_left():
            return deadline - time.time()

        # CMA-ES Parameters
        # Lambda (population size)
        # For dim=5, lambda=10-20 is typical. Let's use 15.
        lam = 15
        mu = lam // 2  # Number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial step size
        # Heuristic: sigma_0 = 1.0 * (max - min) / 4 ? Or just 2.0
        # The box is [-5, 5], width 10.
        sigma = 2.0
        
        # Restart logic
        max_stagnant = 20
        stagnant_count = 0
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_used += 1
            return f_val

        # Main Loop
        while calls_used < budget and time_left() > 10:
            # Initialize CMA-ES state for this "run"
            # We restart from best_x if we have one, otherwise random
            if best_x is None:
                m = np.random.uniform(-5, 5, dim)
            else:
                # Restart from best point but perturb slightly or just use it
                # If we restart from best point, we need to reset C and sigma
                m = best_x.copy()
            
            # Reset CMA-ES state variables
            C = np.eye(dim)
            # Step size adaptation state
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # Initial sigma for restart
            # If we are restarting after stagnation, we might want to reduce sigma
            # or keep it. Let's start with a moderate sigma.
            current_sigma = 2.0 if best_x is None else 1.0
            
            # Run CMA-ES until stagnation or budget/time
            gen = 0
            local_stagnant = 0
            last_best = np.inf
            
            while calls_used < budget and time_left() > 5:
                # Generate population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes non-PD
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                # Sample
                z = np.random.randn(lam, dim)
                # Transform
                # y = m + sigma * (z @ L.T)
                y = m + current_sigma * (z @ L.T)
                
                # Evaluate
                fy = np.array([eval_func(yp) for yp in y])
                
                if calls_used >= budget:
                    break
                
                # Update best
                min_idx = np.argmin(fy)
                if fy[min_idx] < best_f:
                    best_f = fy[min_idx]
                    best_x = y[min_idx].copy()
                    local_stagnant = 0
                else:
                    local_stagnant += 1
                    
                # Sort indices
                idx = np.argsort(fy)
                y_sorted = y[idx]
                fy_sorted = fy[idx]
                
                # Update mean
                m_new = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
                
                # Update p_sigma
                # c_sigma
                c_sigma = (2 + math.log(mu_eff + 2)) / (dim + 4)
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / current_sigma * (L.T @ (m_new - m))
                
                # Update sigma
                # Heaviside function
                h_sigma = 1 if (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1) ) else 0
                current_sigma *= math.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
                
                # Update p_c
                c_c = (4 + math.log(dim) - math.log(mu_eff)) / (dim + 4)
                h_c = 1 if (np.linalg.norm(p_c) / math.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) < (1.7 + 2 / (dim + 1)) * math.sqrt(mu_eff + 1 - 1) ) else 0
                p_c = (1 - c_c) * p_c + h_c * math.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / current_sigma
                
                # Update C
                c1 = 2 / ((dim + 1.3)**2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                
                # Rank-one update
                C = (1 + c1 * (h_c - 1) + cmu) * C - c1 * np.outer(p_c, p_c)
                
                # Rank-mu update
                y_inv = (y_sorted[:mu] - m) / current_sigma
                # C += cmu * sum w_i * y_i y_i^T
                # Note: y_inv is shape (mu, dim)
                C += cmu * (w[:, np.newaxis] * (y_inv * y_inv)).sum(axis=0)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Check for positive definiteness / eigenvalues
                # If C becomes bad, reset
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.any(eigvals < 1e-10):
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        current_sigma = 1.0
                except:
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    current_sigma = 1.0

                m = m_new
                gen += 1
                
                # Stagnation check for this CMA-ES run
                if local_stagnant > 30:
                    break
                if gen > 50: # Max generations per run to avoid infinite loops in bad spots
                    break

        # Final Local Refinement with Nelder-Mead
        if best_x is not None and calls_used < budget and time_left() > 10:
            # Simple Nelder-Mead implementation
            def nm_func(x):
                nonlocal calls_used
                x = np.clip(x, -5, 5)
                f_val = func(x)
                calls_used += 1
                return f_val

            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                x_temp = best_x.copy()
                # Perturb one dimension
                step = 0.1 if abs(best_x[i]) < 1 else 0.01
                x_temp[i] += step
                simplex.append(x_temp)
            
            f_values = [nm_func(x) for x in simplex]
            
            # Nelder-Mead loop
            max_iter = 100
            for _ in range(max_iter):
                if calls_used >= budget or time_left() < 2:
                    break
                    
                # Sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + 1.0 * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = nm_func(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + 2.0 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = nm_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = nm_func(x_c)
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_values[i] = nm_func(simplex[i])
                
                # Update global best
                min_idx = np.argmin(f_values)
                if f_values[min_idx] < best_f:
                    best_f = f_values[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
