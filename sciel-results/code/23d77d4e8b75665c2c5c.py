import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        x_best = None
        f_best = np.inf
        
        rng = np.random.default_rng(12345)
        
        used_calls = 0
        
        # CMA-ES Parameters
        # For dim=5, standard N is 10. Let's use N=10 or 12.
        N = 10
        mu = N // 2
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Restart strategy
        # The "Attractive Sector" function (f6) is known to have a narrow valley.
        # CMA-ES is good at finding it if started in the right basin.
        # We will use a multi-start strategy.
        
        # Initial exploration: a few random points to see the landscape scale.
        # But CMA-ES needs a starting mean.
        # Let's try a few short runs, or one long run with restarts on stagnation.
        
        # Given budget 500 and N=10, we have ~50 generations per run.
        # If we do 10 runs of 50 evals each, that's 500 evals.
        # Or 5 runs of 100 evals.
        
        # Strategy:
        # 1. Pick a random start.
        # 2. Run CMA-ES until:
        #    - Budget is low
        #    - Stagnation (no improvement for X generations)
        #    - Convergence (sigma very small)
        # 3. Restart from a new random point (or slightly perturbed best).
        
        max_generations_per_run = 40  # 400 evals per run max
        min_improvement_threshold = 1e-6
        
        while used_calls < total_budget and time.time() < deadline:
            # Choose starting point
            # Random uniform in [-5, 5]
            x_start = rng.uniform(-5, 5, dim)
            
            # CMA-ES State
            mean = x_start.copy()
            sigma = 1.0  # Initial step size
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Evaluate starting point
            if used_calls >= total_budget:
                break
            f_mean = func(mean)
            used_calls += 1
            
            if f_mean < f_best:
                f_best = f_mean
                x_best = mean.copy()
            
            best_run_f = f_mean
            best_run_x = mean.copy()
            no_improve_count = 0
            stagnation_limit = 15  # Generations without improvement
            
            for gen in range(max_generations_per_run):
                if used_calls + N > total_budget or time.time() > deadline:
                    break
                if sigma < 1e-12:
                    break
                    
                # Check stagnation
                if no_improve_count > stagnation_limit:
                    break
                    
                # Sample offspring
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    
                B = rng.standard_normal((N, dim)) @ L.T
                Y = mean + sigma * B
                Y = np.clip(Y, -5, 5)
                
                # Evaluate offspring
                F = np.empty(N)
                for i in range(N):
                    F[i] = func(Y[i])
                    used_calls += 1
                    if F[i] < f_best:
                        f_best = F[i]
                        x_best = Y[i].copy()
                    if F[i] < best_run_f:
                        best_run_f = F[i]
                        best_run_x = Y[i].copy()
                        
                # Sort and select
                I = np.argsort(F)
                Y_sorted = Y[I]
                F_sorted = F[I]
                
                # Update no_improve_count
                if F_sorted[0] < best_run_f - min_improvement_threshold:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Update mean
                mean_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update sigma and p_sigma
                y_sigma = (mean_new - mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 2))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 2))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = (Y_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                    
                C = (1 + c_1 * h_c - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                mean = mean_new
                
                # Check convergence
                if np.linalg.norm(mean - mean_new) < 1e-12:
                    break
                    
            # End of run
            # If we have budget left, loop and restart

        return x_best, f_best
