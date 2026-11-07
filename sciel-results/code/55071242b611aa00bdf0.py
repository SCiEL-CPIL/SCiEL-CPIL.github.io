import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        best_x = np.zeros(n)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial exploration to find a good starting point
        # Sample several points to bracket the optimum
        rng = np.random.default_rng(12345)
        
        # Try origin and a few random points
        candidates = [np.zeros(n)]
        for _ in range(8):
            candidates.append(rng.uniform(-4, 4, n))
        
        for c in candidates:
            fc = func(c)
            update_best(c, fc)
        
        if best_f == np.inf:
            return best_x, best_f

        # Phase 2: CMA-ES with restarts
        # Parameters tuned for 5 dimensions
        lam = 8  # population size
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Multiple restarts to handle potential local minima or poor initial sigma
        max_restarts = 5
        budget_per_restart = budget // max_restarts
        remaining_budget = budget
        
        for restart in range(max_restarts):
            if remaining_budget < lam:
                break
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES from current best
            x_mean = best_x.copy()
            sigma = 0.5 * (1.0 + 0.2 * restart)  # Start with moderate sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            n_gen_this_restart = min(budget_per_restart, remaining_budget) // lam
            if n_gen_this_restart < 1:
                n_gen_this_restart = 1
            
            for g in range(n_gen_this_restart):
                if time.time() > deadline:
                    break
                if remaining_budget < lam:
                    break
                
                # Sample population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.maximum(vals, 1e-10)
                    L = (vecs * np.sqrt(vals)) @ vecs.T
                
                ys = np.zeros((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    ys[i] = x_mean + sigma * (L @ z)
                    ys[i] = np.clip(ys[i], -5, 5)
                
                # Evaluate
                fs = np.array([func(y) for y in ys])
                remaining_budget -= lam
                
                order = np.argsort(fs)
                update_best(ys[order[0]], fs[order[0]])
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * (ys[order[i]] - x_mean) / sigma
                
                # Update paths
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
                
                # Update sigma
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 5.0)
                
                # Update mean
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5, 5)
                
                # Update covariance
                h_sig = 1.0 if norm_ps < (1.4 + 1 / c_sigma) * chi_n else 0.0
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = (ys[order[i]] - x_mean) / sigma
                    rank_mu += w[i] * np.outer(diff, diff)
                
                C = (1 - c_1 - c_mu) * C + \
                    c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_sigma * (2 - c_sigma) * (np.outer(ps, ps) - np.trace(C) * np.eye(n))) + \
                    c_mu * rank_mu
                C = (C + C.T) / 2
                
                # Check for convergence
                if sigma < 1e-8:
                    break
            
            # After each restart, if we haven't improved, try a different initial point
            if restart < max_restarts - 1 and remaining_budget >= lam:
                # Perturb the best point for next restart
                perturb = rng.uniform(-0.5, 0.5, n)
                new_start = np.clip(best_x + perturb, -5, 5)
                new_f = func(new_start)
                remaining_budget -= 1
                update_best(new_start, new_f)

        # Phase 3: Use remaining budget for fine-tuning with (1+1) CMA-ES or simple hill climbing
        if remaining_budget > 0 and time.time() < deadline:
            # Simple local search around best point
            step_size = 0.1
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                # Try a random step in a random direction
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                trial = np.clip(best_x + step_size * direction, -5, 5)
                f_trial = func(trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    step_size *= 1.1
                else:
                    step_size *= 0.9
                    step_size = max(step_size, 1e-6)

        return best_x, best_f
