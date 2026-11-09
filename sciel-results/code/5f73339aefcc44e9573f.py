import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        x_best = None
        f_best = np.inf
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        
        def check_deadline():
            return used_calls >= total_budget or time.time() > deadline

        # CMA-ES Parameters
        # N = population size
        # mu = number of selected individuals
        N = 10
        mu = N // 2
        
        # Pre-compute weights
        # Standard CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables for CMA-ES
        # We will manage these within the loop, resetting on restart
        
        # Strategy: 
        # 1. Initial broad exploration with CMA-ES centered at 0.
        # 2. If it stagnates, restart centered at best point with smaller sigma.
        # 3. If it stagnates again, random restart.
        
        max_generations_per_run = 20
        min_improvement = 1e-10
        
        run_id = 0
        stagnation_count = 0
        last_best_f = np.inf
        
        while not check_deadline():
            run_id += 1
            
            # Initialize CMA-ES state for this run
            if run_id == 1:
                m = np.zeros(dim)
                sigma = 2.0  # Broad initial search
            else:
                # Restart from best point found so far
                m = x_best.copy()
                # Decay sigma for local refinement
                sigma = 0.5 if run_id == 2 else 0.2 * (run_id - 1) ** 0.5
                if sigma < 0.01:
                    sigma = 0.01

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Run CMA-ES for this phase
            gen = 0
            no_improve_in_phase = 0
            phase_best_f = f_best
            
            while not check_deadline():
                if gen >= max_generations_per_run:
                    break
                
                gen += 1
                
                # Sample population
                # Cholesky factorization of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset C
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Generate N samples
                Z = rng.standard_normal((N, dim))
                X = np.zeros((N, dim))
                for i in range(N):
                    X[i] = m + sigma * (Z[i] @ L.T)
                
                # Evaluate population
                F = np.zeros(N)
                for i in range(N):
                    x_trial = np.clip(X[i], -5, 5)
                    F[i] = func(x_trial)
                    used_calls += 1
                    
                    if F[i] < f_best:
                        f_best = F[i]
                        x_best = x_trial.copy()
                    
                    if used_calls >= total_budget:
                        break
                
                if used_calls >= total_budget:
                    break
                
                # Sort by fitness
                idx = np.argsort(F)
                X_sel = X[idx[:mu]]
                F_sel = F[idx[:mu]]
                
                # Update mean
                m_new = np.sum(weights[:, np.newaxis] * X_sel, axis=0)
                
                # Update step size (CSA)
                # Standard CMA-ES update
                # Calculate expected norm of standard normal
                c_sigma_norm = np.sqrt(dim * (1 - np.exp(-2 / (dim * c_sigma)))) # Approximation for norm
                # Actually, standard formula:
                # E[||N(0, I)||] = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
                # For simplicity, use the standard CMA-ES code logic
                # Here we use a simplified CSA update which is often sufficient for d=5
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (m_new - m))
                
                # Let's stick to the standard update
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (m_new - m))
                
                # Delta term
                delta = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) - 1) * c_sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / c_sigma_norm - 1))
                
                # Update covariance
                # Rank-1 update
                y = (m_new - m) / sigma
                C = (1 + c_1 * delta - c_1) * C + c_1 * np.outer(y, y)
                
                # Rank-mu update
                # This is computationally heavier, but for d=5 it's fine
                # Y = (X_sel - m) / sigma
                # C = (1 + c_mu * delta - c_1 - c_mu) * C + c_mu * np.sum(weights[:, np.newaxis] ** 2 * np.outer(Y.T, Y.T) / mu_eff, axis=2)
                
                # For performance and stability, we might skip rank-mu in a simple implementation 
                # or implement it carefully. Let's include it for robustness.
                Y = (X_sel - m) / sigma
                # weights squared normalized
                w2 = weights ** 2 / mu_eff
                C = (1 + c_mu * delta - c_1 - c_mu) * C + c_mu * np.sum(w2[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', Y, Y), axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization to prevent singularity
                C += 1e-6 * np.eye(dim)
                
                # Update mean
                m = m_new
                
                # Check for improvement in this phase
                current_phase_best = np.min(F)
                if current_phase_best < phase_best_f - min_improvement:
                    phase_best_f = current_phase_best
                    no_improve_in_phase = 0
                else:
                    no_improve_in_phase += 1
                    
                # Stagnation check within phase
                if no_improve_in_phase > 5:
                    break
                if sigma < 1e-6:
                    break
                
                # Check global stagnation
                if f_best < last_best_f - min_improvement:
                    last_best_f = f_best
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                if stagnation_count > 10:
                    # Force restart
                    break

        return x_best, f_best
