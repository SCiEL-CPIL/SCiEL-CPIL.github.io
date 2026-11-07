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

        # Phase 1: Initial coarse search with CMA-ES
        # Use a larger population for better exploration
        lambda_ = 20
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initialize CMA-ES
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 3.0  # Start large to cover the box
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff/n - 1) + 1) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Run CMA-ES until we've used ~70% of budget or found a good solution
        target_calls = int(self.budget * 0.7)
        
        while calls < target_calls:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += np.eye(n) * 1e-6
                L = np.linalg.cholesky(C)
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)
            
            # Evaluate
            f_vals = []
            for x in pop:
                f = try_eval(x)
                if f is None:
                    f_vals.append(np.inf)
                else:
                    f_vals.append(f)
            
            # Sort
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update mean
            new_mean = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
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
        
        # Phase 2: Local refinement around the best point
        # Use a smaller step size and more iterations
        if best_x is not None and calls < self.budget:
            local_mean = best_x.copy()
            local_sigma = 0.5
            local_lambda = 10
            local_mu = local_lambda // 2
            local_weights = np.log(local_mu + 0.5) - np.log(np.arange(1, local_mu + 1))
            local_weights /= local_weights.sum()
            local_mu_eff = 1.0 / np.sum(local_weights**2)
            
            local_c_sigma = (local_mu_eff + 2) / (n + local_mu_eff + 5)
            local_d_sigma = 1 + 2 * max(0, np.sqrt(local_mu_eff/n - 1)) + local_c_sigma
            local_c_c = (4 + local_mu_eff/n) / (n + 4 + 2*local_mu_eff/n)
            local_c_1 = 2 / ((n + 1.3)**2 + local_mu_eff)
            local_c_mu = min(1 - local_c_1, 2 * (np.sqrt(local_mu_eff/n - 1) + 1) / ((n + 2)**2 + local_mu_eff))
            
            local_p_sigma = np.zeros(n)
            local_p_c = np.zeros(n)
            local_C = np.eye(n)
            
            while calls < self.budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = []
                try:
                    L = np.linalg.cholesky(local_C)
                except np.linalg.LinAlgError:
                    local_C += np.eye(n) * 1e-6
                    L = np.linalg.cholesky(local_C)
                
                for i in range(local_lambda):
                    z = rng.standard_normal(n)
                    x_new = local_mean + local_sigma * (L @ z)
                    x_new = np.clip(x_new, -5, 5)
                    pop.append(x_new)
                
                # Evaluate
                f_vals = []
                for x in pop:
                    f = try_eval(x)
                    if f is None:
                        f_vals.append(np.inf)
                    else:
                        f_vals.append(f)
                
                # Sort
                order = np.argsort(f_vals)
                sorted_pop = [pop[i] for i in order]
                sorted_f = [f_vals[i] for i in order]
                
                # Update mean
                new_mean = np.sum([local_weights[i] * sorted_pop[i] for i in range(local_mu)], axis=0)
                
                # Update paths
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)
                
                local_p_sigma = (1 - local_c_sigma) * local_p_sigma + local_c_sigma * np.sqrt(local_mu_eff) * (C_inv_sqrt @ (new_mean - local_mean) / local_sigma)
                
                chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
                h_sigma = 1 if np.linalg.norm(local_p_sigma) / np.sqrt(1 - (1 - local_c_sigma)**(2 * (calls / local_lambda))) < (1.4 + 2 / (n + 1)) * chi_n else 0
                
                local_p_c = (1 - local_c_c) * local_p_c + h_sigma * local_c_c * np.sqrt(local_mu_eff) * (new_mean - local_mean) / local_sigma
                
                # Update covariance
                v = (new_mean - local_mean) / local_sigma
                local_C = (1 + local_c_1 * (h_sigma * (1 + (1 - local_c_sigma) * local_c_sigma) - local_c_1) - local_c_mu) * local_C \
                    + local_c_1 * (np.outer(v, v) + (1 - h_sigma) * local_c_sigma * (1 - local_c_sigma) * local_C)
                
                y = np.array([(sorted_pop[i] - local_mean) / local_sigma for i in range(local_mu)])
                local_C += local_c_mu * np.sum([local_weights[i] * np.outer(y[i], y[i]) for i in range(local_mu)], axis=0)
                
                local_C = (local_C + local_C.T) / 2
                
                # Update sigma
                local_sigma = local_sigma * np.exp(local_c_sigma / local_d_sigma * (np.linalg.norm(local_p_sigma) / chi_n - 1))
                
                # Update mean
                local_mean = new_mean
                
                # Early stopping if convergence
                if local_sigma < 1e-6:
                    break
        
        if best_x is None:
            best_x = np.zeros(n)
            try_eval(best_x)
        
        return best_x, best_f
