import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        # mu: number of best individuals used for update
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial parameters
        x_mean = np.zeros(d)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix C
        C = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Update rates
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (d + 1) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Normalization of C
        # In standard CMA-ES, C is updated via rank-one and rank-mu updates
        # We need to ensure C remains positive definite and symmetric
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Pre-allocate arrays for population
        pop_x = np.zeros((lam, d))
        pop_f = np.zeros(lam)
        
        # Loop until budget or deadline
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset or fix
                    C = np.eye(d)
                    L = np.eye(d)
                
                z = np.random.randn(d)
                # Transform z to have covariance C
                y = x_mean + sigma * (L @ z)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                
                pop_x[i] = y
                pop_f[i] = func(y)
                calls += 1
                
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = y.copy()
                
                # If we hit deadline mid-population, break
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
            
            # Sort population by fitness
            idx = np.argsort(pop_f)
            pop_x_sorted = pop_x[idx]
            pop_f_sorted = pop_f[idx]
            
            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_x_sorted[0].copy()
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * pop_x_sorted[:mu], axis=0)
            
            # Update sigma
            # Calculate normalized step
            step = (x_mean - x_old) / sigma
            step_norm = np.linalg.norm(step)
            
            # Update p_sigma
            h_sigma = 1.0 if step_norm < (1.4 + 2.0 / (d + 1)) * np.sqrt((1 - np.exp(-2.0 / c_sigma)) / (1 - np.exp(-2.0 / c_sigma * lam))) else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * step
            
            # Update p_c
            h_c = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - np.exp(-2 * calls / c_sigma))) < (1.4 + 2.0 / (d + 1)) * np.sqrt(1 - np.exp(-2.0 / c_c)) else 0.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * step
            
            # Update C
            # Rank-one update
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C - C)
            
            # Rank-mu update
            # y_i = (x_i - x_old) / sigma
            Y = (pop_x_sorted[:mu] - x_old) / sigma
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ik,jk->ijk', Y, Y), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            # Standard CMA-ES sigma update uses the ratio of norms
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / sigma_norm - 1))
            # But we need to handle the clipping bias? 
            # For simplicity and robustness, we use the standard update but monitor sigma.
            
            # Check for convergence/stagnation
            # If sigma becomes too small, we might be stuck.
            # Also, if the best fitness hasn't improved for a long time, restart?
            
            # Simple stagnation check: if best_f hasn't improved in last 50 calls, 
            # and sigma is small, do a local restart.
            # For now, let's just let it run. The clipping might slow it down.
            
            # Optional: Polish with Nelder-Mead if sigma is very small
            if sigma < 1e-4 and calls < budget - 10:
                # Use remaining budget for local refinement
                break

        # If we have budget left, do a quick local refinement (Nelder-Mead)
        if calls < budget:
            # Simple Nelder-Mead implementation
            nm_x = best_x.copy()
            nm_f = best_f
            
            # Initial simplex
            simplex = [nm_x.copy()]
            for i in range(d):
                point = nm_x.copy()
                # Step size for simplex
                step = 0.5 if abs(nm_x[i]) > 1e-3 else 0.5
                point[i] += step
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            simplex_f = [nm_f]
            for i in range(1, d + 1):
                simplex_f.append(func(simplex[i]))
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
            
            # Nelder-Mead loop
            while calls < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + 2 * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        break
                    
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
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                break
                        if calls >= budget or time.time() > deadline:
                            break

        return best_x, best_f
