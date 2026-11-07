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
        budget = self.budget
        
        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        history_best = []
        
        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def record_best():
            history_best.append(best_f)

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 12
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        c_hs = 1.1 + 2.0 / (dim + 1.7)
        
        stagnation_limit = 30
        max_evals_per_run = 150
        
        # Multi-Start Loop
        run_id = 0
        while eval_count < budget and not time_check():
            # Initialize CMA-ES state
            # Start point: Random in [-5, 5]
            x_mean = rng.uniform(-5, 5, dim)
            sigma = 1.5 # Larger initial sigma for broader exploration
            
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            generation = 0
            
            # Evaluate initial mean
            f_mean = wrapped_func(x_mean)
            record_best()
            
            best_f_in_run = best_f
            last_improvement_count = 0
            
            while eval_count < budget and not time_check():
                # Check stagnation
                if last_improvement_count >= stagnation_limit:
                    break
                
                # Generate population
                ys = [B @ (D * rng.standard_normal(dim)) for _ in range(lam)]
                xs = x_mean + sigma * np.array(ys)
                
                # Evaluate population
                num_evals = min(lam, budget - eval_count)
                if num_evals <= 0: break
                
                fitnesses = []
                eval_xs = []
                for i in range(num_evals):
                    f = wrapped_func(xs[i])
                    fitnesses.append(f)
                    eval_xs.append(xs[i])
                    record_best()
                    if eval_count >= budget: break
                
                if num_evals < lam: break
                
                fitnesses = np.array(fitnesses)
                order = np.argsort(fitnesses)
                xs_sorted = np.array(eval_xs)[order]
                ys_sorted = np.array(ys)[order][:mu]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                y_mean = (new_mean - x_mean) / sigma
                
                # Update sigma
                norm_y_mean = np.linalg.norm(y_mean)
                sqrt_trace_C = np.sqrt(np.trace(C))
                ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
                
                p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
                norm_p_sigma = np.linalg.norm(p_sigma)
                gen = generation + 1
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1.4 + 2 / (dim + 1)) * chi_n)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                generation = gen
                
                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                rank_one = np.outer(p_c, p_c)
                if not h_sigma:
                    rank_one += c_c * (2 - c_c) * C
                
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                x_mean = new_mean
                
                # Eigendecomposition
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.clip(D2, 1e-10, None)
                    D = np.sqrt(D2)
                except:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                
                # Check for improvement
                if best_f < best_f_in_run:
                    best_f_in_run = best_f
                    last_improvement_count = 0
                else:
                    last_improvement_count += 1
                    
                if sigma < 1e-6:
                    break

            run_id += 1
            # If we exited due to budget, the loop condition will handle it
            # If we exited due to stagnation, we restart

        # Fill history
        while len(history_best) < budget:
            history_best.append(history_best[-1] if history_best else best_f)

        return best_x, best_f
