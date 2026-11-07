import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # --- Phase 1: CMA-ES ---
        # Parameters for CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 4 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = dim + 2 + c_sigma - 1
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # State
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls_made = 1
        
        stagnation_counter = 0
        stagnation_limit = 15 # Generations without improvement before considering restart or local search
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check for convergence or stagnation to switch to local search earlier
            if sigma < 1e-4 or stagnation_counter > stagnation_limit:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls_made >= budget: break
                if time.time() > deadline: break
                
                # Sample from N(0, C)
                C = (C + C.T) / 2.0
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                
                z = rng.standard_normal(dim)
                y = L @ z
                x_trial = m + sigma * y
                x_trial_clipped = np.clip(x_trial, -5, 5)
                
                f_val = func(x_trial_clipped)
                calls_made += 1
                
                ys.append(y)
                xs.append(x_trial_clipped)
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_trial_clipped.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            
            if not ys:
                break
            
            # Sort by fitness
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            m_new = m + sigma * np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            
            # Update sigma
            try:
                C_inv_sqrt_m = np.linalg.solve(L, (m_new - m) / sigma)
            except:
                C_inv_sqrt_m = (m_new - m) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_m
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * chi_N else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt_m
            
            # Update C
            y_rank1 = np.outer(p_c, p_c)
            if h_sigma == 0:
                y_rank1 += c_c * (2 - c_c) * C
            
            y_rankmu = np.zeros((dim, dim))
            for i in range(mu):
                y_rankmu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * y_rank1 + c_mu * y_rankmu
            C = (C + C.T) / 2.0
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            m = m_new
            m = np.clip(m, -5, 5)
            
        # --- Phase 2: Local Search (Coordinate Descent with Perturbations) ---
        # Use remaining budget for local refinement
        if calls_made < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Simple Coordinate Descent
            step_size = 0.1
            no_improve_count = 0
            max_no_improve = 20
            
            while calls_made < budget and time.time() < deadline and no_improve_count < max_no_improve:
                improved = False
                for i in range(dim):
                    # Try positive step
                    x_trial = x_local.copy()
                    x_trial[i] += step_size
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    
                    if f_trial < f_local:
                        x_local = x_trial
                        f_local = f_trial
                        improved = True
                    else:
                        # Try negative step
                        x_trial = x_local.copy()
                        x_trial[i] -= step_size
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_made += 1
                        
                        if f_trial < f_local:
                            x_local = x_trial
                            f_local = f_trial
                            improved = True
                    
                    if calls_made >= budget or time.time() > deadline:
                        break
                
                if not improved:
                    # Reduce step size
                    step_size *= 0.5
                    no_improve_count += 1
                    
                    # Add a small random perturbation to escape local minima
                    if step_size > 1e-6 and no_improve_count % 5 == 0:
                        x_perturbed = x_local + rng.standard_normal(dim) * step_size
                        x_perturbed = np.clip(x_perturbed, -5, 5)
                        f_perturbed = func(x_perturbed)
                        calls_made += 1
                        if f_perturbed < f_local:
                            x_local = x_perturbed
                            f_local = f_perturbed
                            improved = True
                            no_improve_count = 0
                else:
                    no_improve_count = 0
                    
                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()
        
        return best_x, best_f
