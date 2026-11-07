import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        
        n = self.dim
        best_x = None
        best_f = np.inf
        calls = 0

        rng = np.random.default_rng(42)

        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global CMA-ES with Restart on Stagnation
        # We allocate 60% of the budget to global search
        target_calls_global = int(self.budget * 0.6)
        
        # CMA-ES Parameters
        lambda_ = 15
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff/n - 1) + 1) / ((n + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
        
        # Initial State
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 3.0
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        no_improve_count = 0
        best_f_this_run = np.inf
        
        # Run Global Phase
        while calls < target_calls_global and time.time() < deadline:
            # Restart if stuck
            if no_improve_count > 10 and calls > 50:
                if best_x is not None:
                    mean = best_x.copy()
                    sigma *= 0.5
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    no_improve_count = 0
                    best_f_this_run = best_f
                else:
                    # If we haven't found anything, just keep going
                    no_improve_count = 0
                    best_f_this_run = best_f

            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += np.eye(n) * 1e-6
                L = np.linalg.cholesky(C)
            
            pop = []
            f_vals = []
            
            for i in range(lambda_):
                if calls >= target_calls_global or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                f = try_eval(x_new)
                if f is None:
                    f_vals.append(np.inf)
                    pop.append(x_new)
                else:
                    f_vals.append(f)
                    pop.append(x_new)
            
            if len(f_vals) < lambda_:
                # Pad with inf if we ran out of budget mid-generation
                while len(f_vals) < lambda_:
                    f_vals.append(np.inf)
                    pop.append(pop[-1] if pop else np.zeros(n))
            
            # Sort
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update no_improve
            if sorted_f[0] < best_f_this_run - 1e-10:
                best_f_this_run = sorted_f[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Update mean
            new_mean = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (new_mean - mean) / sigma
            
            # Update covariance
            v = (new_mean - mean) / sigma
            C = (1 + c_1 * (h_sigma * (1 + (1 - c_sigma) * c_sigma) - c_1) - c_mu) * C \
                + c_1 * (np.outer(v, v) + (1 - h_sigma) * c_sigma * (1 - c_sigma) * C)
            
            y = np.array([(sorted_pop[i] - mean) / sigma for i in range(mu)])
            C += c_mu * np.sum([weights[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Early stopping if convergence
            if sigma < 1e-6:
                break

        # Phase 2: Local Refinement with 1+1 CMA-ES
        # Use remaining budget for fine-grained search around best_x
        if best_x is not None and calls < self.budget:
            local_mean = best_x.copy()
            local_sigma = 0.3 # Start small
            local_C = np.eye(n)
            local_p_sigma = np.zeros(n)
            local_p_c = np.zeros(n)
            
            # 1+1 Parameters
            mu_w = 1
            c_sigma_11 = (mu_w + 2) / (n + mu_w + 5)
            d_sigma_11 = 1 + 2 * max(0, np.sqrt(mu_w/n - 1)) + c_sigma_11
            c_c_11 = (4 + mu_w/n) / (n + 4 + 2*mu_w/n)
            c_1_11 = 2 / ((n + 1.3)**2 + mu_w)
            c_mu_11 = min(1 - c_1_11, 2 * (np.sqrt(mu_w/n - 1) + 1) / ((n + 2)**2 + mu_w))
            
            chi_n_11 = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            while calls < self.budget:
                if time.time() > deadline:
                    break
                
                try:
                    L = np.linalg.cholesky(local_C)
                except np.linalg.LinAlgError:
                    local_C += np.eye(n) * 1e-6
                    L = np.linalg.cholesky(local_C)
                
                z = rng.standard_normal(n)
                x_new = local_mean + local_sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                
                f_new = try_eval(x_new)
                if f_new is None:
                    break
                
                # Update paths
                y_norm = (x_new - local_mean) / local_sigma
                local_p_sigma = (1 - c_sigma_11) * local_p_sigma + np.sqrt(c_sigma_11 * (2 - c_sigma_11) * mu_w) * y_norm
                
                h_sigma = 1 if np.linalg.norm(local_p_sigma) / np.sqrt(1 - (1 - c_sigma_11)**(calls + 1)) < (1.4 + 2 / (n + 1)) * chi_n_11 else 0
                
                local_p_c = (1 - c_c_11) * local_p_c + h_sigma * np.sqrt(c_c_11 * (2 - c_c_11) * mu_w) * y_norm
                
                if f_new < best_f:
                    # Success: Update mean and covariance
                    local_mean = x_new.copy()
                    
                    # Update covariance
                    local_C = (1 + c_1_11 * (h_sigma * (1 + (1 - c_sigma_11) * c_sigma_11) - c_1_11) - c_mu_11) * local_C \
                        + c_1_11 * (np.outer(local_p_c, local_p_c) + (1 - h_sigma) * c_sigma_11 * (1 - c_sigma_11) * local_C)
                    local_C += c_mu_11 * np.outer(y_norm, y_norm)
                    local_C = (local_C + local_C.T) / 2
                    
                    # Adapt sigma
                    local_sigma = local_sigma * np.exp(c_sigma_11 / d_sigma_11 * (np.linalg.norm(local_p_sigma) / chi_n_11 - 1))
                    
                    # Clip sigma
                    local_sigma = np.clip(local_sigma, 1e-6, 5.0)
                    
                    if local_sigma < 1e-7:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            try_eval(best_x)
        
        return best_x, best_f
