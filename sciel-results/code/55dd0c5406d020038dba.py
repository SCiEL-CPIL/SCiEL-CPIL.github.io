"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizer using Diagonal CMA-ES with PCA initialization for BBOB f18 (Schaffers F7 ill-conditioned).
    
    Key ideas:
    1. Diagonal covariance (per-dimension step sizes) for numerical stability and speed
    2. PCA-based initialization from early samples to align with the funnel structure
    3. Adaptive restarts when stagnation is detected
    4. Efficient budget management with deadline guards
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        dim = self.dim
        total_budget = self.budget
        
        # Diagonal CMA-ES Parameters
        lambda_pop = 4 + int(3 * math.log(dim))  # 9 for dim=5
        if lambda_pop < 4:
            lambda_pop = 4
            
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Update rates for diagonal CMA-ES
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 2.0 / mueff) / ((dim + 2.0)**2 + mueff))
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, math.sqrt((mu - 1) / (dim + 1)) - 1.0) + cs
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        def time_check():
            return time.time() < deadline - 2.0
        
        def evaluate_point(x):
            nonlocal calls, best_f, best_x
            if not time_check():
                return None
            if calls >= total_budget:
                return None
            try:
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except:
                return None
        
        # Phase 1: Initial random sampling to estimate the landscape orientation
        n_init = min(20, total_budget // 5)
        init_points = []
        init_fvals = []
        
        for i in range(n_init):
            if not time_check() or calls >= total_budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = evaluate_point(x)
            if f is not None:
                init_points.append(x)
                init_fvals.append(f)
        
        # Phase 2: PCA-based initialization if we have enough samples
        if len(init_points) >= dim + 1:
            init_points_arr = np.array(init_points)
            init_fvals_arr = np.array(init_fvals)
            
            # Weight by inverse fitness (lower fitness = higher weight)
            f_min = np.min(init_fvals_arr)
            f_range = np.max(init_fvals_arr) - f_min + 1e-10
            w = np.exp(-3.0 * (init_fvals_arr - f_min) / f_range)
            w /= np.sum(w)
            
            # Compute weighted covariance
            mean_init = np.sum(w[:, np.newaxis] * init_points_arr, axis=0)
            centered = init_points_arr - mean_init
            cov_init = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(centered, centered), axis=0)
            
            # PCA to get principal directions
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_init)
                # Sort by eigenvalue (descending)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                
                # Initialize step sizes based on eigenvalues
                eigvals = np.maximum(eigvals, 1e-10)
                sigma_init = np.sqrt(eigvals)
                sigma_init = np.clip(sigma_init, 0.1, 5.0)
            except:
                sigma_init = np.ones(dim) * 2.0
                mean_init = np.zeros(dim)
        else:
            # Fallback: uniform initialization
            sigma_init = np.ones(dim) * 2.0
            mean_init = np.zeros(dim)
        
        # Phase 3: Diagonal CMA-ES
        while calls < total_budget and time_check():
            # Initialize or reset CMA-ES state
            mean = mean_init.copy()
            sigma_diag = sigma_init.copy()
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Track progress for restart decision
            best_f_in_run = float('inf')
            calls_in_run = 0
            no_improvement_calls = 0
            max_no_improvement = 80
            max_calls_per_run = 150
            
            while calls < total_budget and calls_in_run < max_calls_per_run and time_check():
                # Sample population using diagonal covariance
                z = np.random.randn(lambda_pop, dim)
                pop = mean[np.newaxis, :] + sigma_diag[np.newaxis, :] * z
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if calls >= total_budget or not time_check():
                        break
                    f = evaluate_point(x)
                    if f is not None:
                        fvals.append(f)
                        calls_in_run += 1
                        if f < best_f_in_run:
                            best_f_in_run = f
                            no_improvement_calls = 0
                        else:
                            no_improvement_calls += 1
                    else:
                        fvals.append(float('inf'))
                        calls_in_run += 1
                
                if len(fvals) < lambda_pop:
                    fvals.extend([float('inf')] * (lambda_pop - len(fvals)))
                
                # Sort population
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = np.array(fvals)[order]
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Check for convergence
                if np.all(np.abs(new_mean - mean) < 1e-12):
                    break
                
                # Update evolution paths
                sigma_norm = np.linalg.norm(sigma_diag)
                if sigma_norm > 1e-15:
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / sigma_norm * (new_mean - mean) / sigma_diag
                else:
                    ps = (1 - cs) * ps
                
                hsig = 1 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**((calls + 1) / (cs * (dim + 1)))) < (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1)) else 0
                
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma_diag
                
                # Update diagonal covariance (step sizes)
                # For diagonal CMA-ES, we update each dimension independently
                for i in range(dim):
                    # Rank-mu update for dimension i
                    rank_mu_update = cmu * np.sum(weights * ((sorted_pop[:mu, i] - mean[i]) / sigma_diag[i])**2)
                    # Rank-1 update
                    rank_1_update = c1 * (pc[i]**2 - 1.0)
                    
                    # Combine updates
                    log_sigma_update = rank_1_update + rank_mu_update
                    sigma_diag[i] *= math.exp(0.5 * log_sigma_update)
                    
                    # Clip step size
                    sigma_diag[i] = np.clip(sigma_diag[i], 1e-6, 10.0)
                
                # Update sigma (global scale)
                sigma_mean = np.mean(sigma_diag)
                sigma_old_mean = np.mean(sigma_diag)  # This is a simplification
                sigma_diag *= math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(mueff) - 1.0))
                
                # Update mean
                mean = new_mean
                mean = np.clip(mean, -5, 5)
                
                # Check for restart condition
                if no_improvement_calls > max_no_improvement and calls_in_run > 30:
                    break
            
            # If we still have budget, restart from a perturbed best point
            if calls < total_budget and time_check():
                # Perturb the best point to escape local minima
                perturbation_scale = 0.3
                best_x_perturbed = best_x + perturbation_scale * np.random.randn(dim)
                best_x_perturbed = np.clip(best_x_perturbed, -5, 5)
                
                f = evaluate_point(best_x_perturbed)
                if f is not None and f < best_f:
                    # Restart from this new best point
                    mean_init = best_x_perturbed.copy()
                    sigma_init = sigma_diag.copy() * 0.5
                else:
                    # Restart from the original best with smaller step sizes
                    mean_init = best_x.copy()
                    sigma_init = sigma_diag.copy() * 0.5
        
        return best_x, best_f
# EVOLVE-BLOCK-END
