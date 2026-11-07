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
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        rng = np.random.default_rng(42)

        # 1. Initial Global Exploration
        # Spend ~20% of budget on random sampling to find a good starting point
        n_init = max(10, int(budget * 0.2))
        
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. CMA-ES with Restarts
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * math.log(dim)))  # Population size
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        sigma_path = 1 + 2 * max(0, (cs - 1) / (dim + 1)) - cs
        
        # Restart condition
        stagnation_threshold = 1e-6
        stagnation_generations = 10
        calls_since_improve = 0
        best_f_last_restart = best_f
        
        # Main CMA-ES Loop
        while n_calls < budget and time.time() < deadline:
            # Check if we have enough budget for a generation
            if budget - n_calls < lam:
                # Do single-step (1+1) style if budget is low
                while n_calls < budget and time.time() < deadline:
                    # Adaptive step size based on recent improvements
                    step_size = max(0.01, 0.1 * (1 + (best_f_last_restart - best_f) / (1 + abs(best_f_last_restart))))
                    y = np.clip(best_x + step_size * rng.standard_normal(dim), -5.0, 5.0)
                    f_y = timed_call(y)
                    if f_y < best_f:
                        best_f = f_y
                        best_x = y.copy()
                        calls_since_improve = 0
                    else:
                        calls_since_improve += 1
                break

            # Initialize CMA-ES state if this is the first generation or after a restart
            if not hasattr(self, '_cmaes_active') or self._cmaes_active == False:
                mean = best_x.copy()
                sigma = 0.5  # Start with moderate sigma for refinement
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                self._cmaes_active = True
            else:
                mean = self._cmaes_mean
                sigma = self._cmaes_sigma
                C = self._cmaes_C
                pc = self._cmaes_pc
                ps = self._cmaes_ps

            # Generate Population
            Y = []
            FY = []
            # Check time and budget inside loop
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Compute y = mean + sigma * L @ z
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                y = np.clip(mean + sigma * (L @ z), -5.0, 5.0)
                fy = timed_call(y)
                Y.append(y)
                FY.append(fy)
            
            if len(Y) < lam:
                break

            # Sort by fitness
            idx = np.argsort(FY)
            Y_sorted = [Y[i] for i in idx]
            FY_sorted = [FY[i] for i in idx]
            
            # Update Best
            if FY_sorted[0] < best_f:
                best_f = FY_sorted[0]
                best_x = Y_sorted[0].copy()
                calls_since_improve = 0
            else:
                calls_since_improve += lam
            
            # Update CMA-ES Parameters
            # Mean update
            new_mean = np.sum([weights[i] * Y_sorted[i] for i in range(mu)], axis=0)
            
            # Step size control
            # ps = (1-cs)*ps + cs/sqrt(mu_eff) * (C^(-1/2) * (new_mean - mean)/sigma)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals_clipped = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals_clipped)) @ eigvecs.T
                ps = (1 - cs) * ps + cs / np.sqrt(mu_eff) * (C_inv_sqrt @ ((new_mean - mean) / sigma))
            except:
                ps = (1 - cs) * ps + cs / np.sqrt(mu_eff) * ((new_mean - mean) / sigma)

            sigma *= np.exp(cs / sigma_path * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            # Covariance matrix update
            # Update pc
            pc = (1 - c1) * pc + c1 / np.sqrt(mu_eff) * (new_mean - mean) / sigma
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_diff = (Y_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_diff, y_diff)
            
            # Standard CMA-ES covariance update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 + (1 - cs) * (2 - cs)) * cs * (1 - cs) * np.outer(ps, ps)) + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Sometimes numerical errors cause issues, re-regularize
            if not np.all(np.isfinite(C)):
                C = np.eye(dim)
            
            # Save state
            self._cmaes_mean = new_mean
            self._cmaes_sigma = sigma
            self._cmaes_C = C
            self._cmaes_pc = pc
            self._cmaes_ps = ps

            # Check for Stagnation/Restart
            if calls_since_improve >= stagnation_generations * lam:
                # Check if we made *any* significant progress since last restart
                if best_f > best_f_last_restart - stagnation_threshold:
                    # Restart
                    best_f_last_restart = best_f
                    calls_since_improve = 0
                    self._cmaes_active = False  # Force re-initialization in next loop
                    # Optional: Perturb the best point slightly or reset to origin?
                    # Resetting to origin might be good for global exploration.
                    # Let's try a random restart point if we are stuck.
                    x_new = rng.uniform(-5.0, 5.0, dim)
                    f_new = timed_call(x_new) if n_calls < budget else float('inf')
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                    # We will re-init CMA-ES around the best_x
                    continue

        return best_x, best_f
