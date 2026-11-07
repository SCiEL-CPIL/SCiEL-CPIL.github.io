import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Exploration
        # Sample a few points to get a rough sense of the landscape
        n_init = 10
        for _ in range(n_init):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: Portfolio of CMA-ES
        # We run two CMA-ES instances with different initial sigmas.
        # Instance 1: Aggressive (larger sigma)
        # Instance 2: Conservative (smaller sigma)
        
        def run_cmaes(sigma_init, pop_size, seed_offset):
            """
            Runs a CMA-ES instance. Returns the best x and f found by this instance.
            Uses the shared `evaluate` function which tracks global budget and best.
            """
            rng_local = np.random.default_rng(42 + seed_offset)
            
            # CMA-ES Parameters
            mean = best_x.copy() # Start from current global best
            sigma = sigma_init
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            # Learning rates
            c_sigma = (pop_size + 2) / (d + pop_size + 5)
            c_c = (4 + pop_size / d) / (d + 4 + 2 * pop_size / d)
            c_1 = 2 / ((d + 1.3)**2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 - 1 + 1 / d) / (d + 2) - 1))
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (d + 1)) - 1) + c_sigma
            chi_lambda = np.sqrt(pop_size) * (1 - 1/(4*pop_size) + 1/(21*pop_size**2))
            
            mu = pop_size // 2
            log_weights = np.log(np.arange(mu, 0, -1))
            weights = np.exp(log_weights - np.mean(log_weights))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            local_best_f = np.inf
            local_best_x = mean.copy()
            stagnation = 0
            
            while check_time():
                # Generate population
                pop = []
                for i in range(pop_size):
                    if not check_time():
                        break
                    z = rng_local.standard_normal(d)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(d)
                    x_i = mean + sigma * (L @ z)
                    x_i = np.clip(x_i, -5, 5)
                    pop.append(x_i)
                
                if len(pop) < 2:
                    break
                
                # Evaluate
                f_vals = []
                for x_i in pop:
                    fx = evaluate(x_i)
                    if fx is None:
                        f_vals.append(np.inf)
                    else:
                        f_vals.append(fx)
                        if fx < local_best_f:
                            local_best_f = fx
                            local_best_x = x_i.copy()
                
                # Sort
                indices = np.argsort(f_vals)
                
                # Update mean
                new_mean = np.zeros(d)
                for i in range(mu):
                    new_mean += weights[i] * pop[indices[i]]
                
                # Update sigma
                if np.linalg.norm(new_mean - mean) > 1e-12:
                    p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / sigma * (new_mean - mean)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
                else:
                    p_sigma = (1 - c_sigma) * p_sigma
                
                # Update covariance
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / sigma * (new_mean - mean)
                delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(d+1)) * chi_lambda else 0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - delta_h) * C)
                
                for i in range(mu):
                    y_i = (pop[indices[i]] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                
                mean = new_mean
                mean = np.clip(mean, -5, 5)
                
                # Stagnation check
                if f_vals[indices[0]] >= local_best_f - 1e-8:
                    stagnation += 1
                else:
                    stagnation = 0
                
                if stagnation > 20:
                    break

            return local_best_x, local_best_f

        # Run Instance 1: Aggressive
        x1, f1 = run_cmaes(sigma_init=3.0, pop_size=10, seed_offset=1)
        
        # Run Instance 2: Conservative
        # Note: We continue from the global best found so far
        x2, f2 = run_cmaes(sigma_init=1.0, pop_size=10, seed_offset=2)

        # Optionally, run a 3rd instance if time permits, focused on the best found
        if check_time() and n_calls < budget - 20:
             # Quick local polish using a very small sigma CMA-ES or just random jitter
             for _ in range(10):
                 if not check_time():
                     break
                 x = best_x + rng.normal(0, 0.1, d)
                 evaluate(x)

        return best_x, best_f
