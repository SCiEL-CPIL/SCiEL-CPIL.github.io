import numpy as np
import time

class Optimizer:
    """
    Multi-Start CMA-ES for BBOB f15 (Rastrigin rotated).
    
    Strategy:
    1. Use CMA-ES with adaptive step size and covariance matrix.
    2. Implement a restart mechanism: if no improvement is found within a certain 
       window of evaluations, restart the local search from a new random point 
       or a perturbed global best.
    3. This handles the multimodality of Rastrigin better than a single 
       population DE which might get trapped if the population converges 
       to a local minimum early.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Global best tracking
        best_x_global = None
        best_f_global = np.inf

        # CMA-ES parameters
        # Initial step size. The domain is [-5, 5]^5. 
        # A reasonable initial sigma is 2.0 or 3.0 to cover the space quickly.
        sigma = 2.0
        
        # Population size for CMA-ES. 
        # Standard is 4 + 3*ln(N). For N=5, ~10. 
        # Smaller pop (e.g., 5-8) allows for more restarts within budget 500.
        # Let's use a smaller pop to allow frequent restarts.
        pop_size = 5

        # Restart threshold: number of evaluations without improvement
        # If we don't improve in 30 evals, restart.
        # 500 budget / 30 = ~16 restarts. This is a good balance.
        restart_threshold = 30
        
        # We will run multiple "runs" of CMA-ES.
        # Each run initializes a mean vector and covariance matrix.
        
        evals_used = 0
        current_run_evals = 0
        best_f_current_run = np.inf
        best_x_current_run = None
        mean = None
        C = None
        pc = None
        ps = None
        alpha_mu = None
        alpha_sigma = None
        chi_n = 0.0
        mu = 0
        
        # Precompute some CMA-ES constants for efficiency
        def init_cmaes(mean_init):
            nonlocal sigma, C, pc, ps, alpha_mu, alpha_sigma, chi_n, mu
            sigma = 2.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Standard CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            alpha_mu = (2 + 2.0 / (dim + 1.0)) / (dim + 4 + mu_eff)
            alpha_sigma = (alpha_mu / 2) * (dim + 2) / (dim + mu_eff + 2)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            
            # State for this run
            best_f_current_run = np.inf
            best_x_current_run = None
            current_run_evals = 0

        # First initialization
        # Start near center or random? Random is safer for rotated functions.
        # But since we restart, we can start random.
        mean = rng.uniform(-4, 4, size=dim)
        init_cmaes(mean)

        while evals_used < total_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((pop_size, dim))
            fvals = np.zeros(pop_size)
            
            # We need to generate 'pop_size' points.
            # Check if we have enough budget left for a full generation.
            # If not, we might do a partial generation or stop.
            # To keep it simple and robust, if remaining budget < pop_size, 
            # we just do single steps or stop.
            
            remaining = total_budget - evals_used
            if remaining <= 0:
                break
            
            # If we can't afford a full generation, do as many as we can
            # but CMA-ES update is complex with partial pops. 
            # Given budget 500 and pop 5, we have 100 full generations.
            # We'll assume we always have enough for a full pop unless we are at the very end.
            if remaining < pop_size:
                # Just do random sampling for the last few
                for k in range(remaining):
                    if time.time() > deadline: break
                    x = rng.uniform(-5, 5, size=dim)
                    fx = func(x)
                    evals_used += 1
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = x.copy()
                break

            # Standard CMA-ES step
            # 1. Sample
            # We need to update mean, C, sigma before sampling? 
            # No, sample from current distribution, evaluate, then update.
            
            # Sample pop_size points
            for k in range(pop_size):
                if time.time() > deadline:
                    break
                # Sample: mean + sigma * z, where z is normal with covariance C
                # Efficient sampling: mean + sigma * L @ z_std
                # We need Cholesky of C.
                # For small dim (5), Cholesky is cheap.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset C
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                x = mean + sigma * (L @ z)
                
                # Clip to bounds
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                evals_used += 1
                pop[k] = x
                fvals[k] = fx
                
                # Update current run best
                if fx < best_f_current_run:
                    best_f_current_run = fx
                    best_x_current_run = x.copy()
                
                # Update global best
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = x.copy()
            
            if evals_used >= total_budget:
                break

            # 2. Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # 3. Update mean
            # Weights are already defined in init_cmaes scope, but let's recalc if needed
            # Actually, weights depend on mu which is constant.
            # Let's define weights locally for clarity or keep them in closure.
            # I'll define them in the outer scope for access.
            
            # Recalculate weights if not done (they are constant)
            # mu = pop_size // 2
            # weights = ...
            
            # Using the weights from init_cmaes
            # Wait, weights is not in nonlocal scope. Let's just recalculate or pass it.
            # For simplicity, I'll recalculate weights here as it's cheap.
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            
            new_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # 4. Update Covariance Matrix (C)
            # Rank-mu update
            v = (new_mean - mean) / sigma
            pc = (1 - alpha_mu) * pc + alpha_mu * v
            
            # Check if pc is too large
            if np.sqrt(np.sum(pc**2) / (1 - (1 - alpha_mu)**(2 * (evals_used + 1)))) > (1.4 + 2 / (dim + 1)) * chi_n:
                # Reset pc
                pc = np.zeros(dim)
            else:
                # Update C
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + delta_pc * C) + cmu * sum(weights * (xi-mean)(xi-mean).T)
                c1 = (2 + 2.0 / (dim + 1.0)) / ((dim + 1.0) ** 2 + 2)
                cmu = (2 - c1 + 2.0 * np.sum(weights[:2]) ) / ((dim + 2) ** 2 + 2)
                
                # delta_pc is 1 if pc update was made, else 0? 
                # Standard formula: delta_pc = 1 if ||pc||/chi_n > ... else 0
                # Actually, the condition above resets pc. If it didn't reset, we assume the update happened.
                # The term is (1 - (1-alpha_mu)^(2*(evals+1))) in denominator.
                # Let's use the standard rank-mu update formula.
                
                # Correction: The pc update condition is usually:
                # if ||pc||/chi_n > 1.4 + 2/(dim+1):
                #     C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + delta_pc * C) + ...
                #     where delta_pc = 1 - (1 - alpha_mu)^(2*(evals+1))
                # else:
                #     C = (1 - c1 - cmu) * C + c1 * pc * pc.T + ...
                
                # My code above reset pc if condition met. 
                # If it didn't reset, we just add pc*pc.T.
                # But the standard CMA-ES uses the cumulative step size control.
                # Let's stick to a simpler, robust update for small dims.
                
                # Simple Rank-Mu Update
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y = (pop[i] - mean) / sigma
                    C += cmu * weights[i] * np.outer(y, y)
            
            # 5. Update Step Size (Sigma)
            ps = (1 - alpha_sigma) * ps + np.sqrt(alpha_sigma * (2 - alpha_sigma) * mu_eff) / sigma * v
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (cs - 1) / (1 - cs)) + cs
            
            # hsig is 1 if ||ps||/sqrt(1-(1-cs)^2*(evals+1)) < (1.4 + 2/(dim+1)) * chi_n
            # Simplified: just use ps norm
            ps_norm = np.linalg.norm(ps)
            # Approximation for hsig
            hsig = 1.0 if ps_norm < (1.4 + 2.0 / (dim + 1.0)) * chi_n else 0.0
            
            sigma *= np.exp(cs / ds * (ps_norm / chi_n - 1.0))
            
            # Prevent sigma from becoming 0 or inf
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update mean for next iteration
            mean = new_mean
            
            # Check for restart
            current_run_evals += pop_size
            if current_run_evals >= restart_threshold:
                # Restart
                # Where to restart?
                # 1. Random new point
                # 2. Best point found so far (global or local)
                # For Rastrigin, jumping to a new random region is often better 
                # to escape local minima.
                # Let's mix: 50% random, 50% perturbed global best
                
                if rng.random() < 0.5 and best_x_global is not None:
                    # Perturb global best
                    mean = best_x_global + rng.normal(0, 1.0, size=dim)
                    mean = np.clip(mean, -5, 5)
                else:
                    # Random new start
                    mean = rng.uniform(-4, 4, size=dim)
                
                # Reinitialize CMA-ES state for new run
                init_cmaes(mean)
        
        return best_x_global, best_f_global
