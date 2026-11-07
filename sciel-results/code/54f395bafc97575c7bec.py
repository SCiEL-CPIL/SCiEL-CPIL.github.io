"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 280.0  # Safety margin

        # Global best
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Use a fixed seed for reproducibility within the run, but different starts for each restart
        base_rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # For dim=5, a population size of lambda = 4 + floor(3*ln(dim)) = 4 + 12 = 16 is standard.
        # Let's use lambda=16.
        lam = 16
        mu = lam // 2  # top half survive
        
        # We will allocate a fixed budget per restart, or dynamically based on remaining time/budget.
        # Let's do multiple restarts.
        # Each restart gets a chunk of the budget.
        # If we do 5 restarts, each gets ~100 evaluations.
        # Or we can do dynamic allocation: if no improvement for X steps, restart.
        
        # Strategy:
        # 1. Start with a random point.
        # 2. Run CMA-ES for a fixed number of generations (e.g., 10-20) or until stagnation.
        # 3. If stagnation, restart from a new random point, but maybe biased towards the current best?
        #    No, for bi-Rastrigin, the two minima are distinct. Random restarts are better.
        
        # Let's implement a simple CMA-ES loop.
        
        def run_cmaes(x0, budget_for_run):
            nonlocal best_f, best_x
            
            # Initialize CMA-ES state
            # Mean
            mean = x0.copy()
            # Step size (sigma)
            # Start with a moderate step size, e.g., 1.0 or 2.0
            sigma = 1.0
            # Covariance matrix C
            C = np.eye(dim)
            # Inverse covariance matrix
            C_inv = np.eye(dim)
            # Pre-compute Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)
            
            # Learning rates
            cc = (4 + dim/3) / (dim + 4 + 2*dim/dim) # path exponent for evolution path
            c1 = 2 / ((dim + 1.3)**2 + mu)
            cmu = min(1 - c1, 2 * (mu - 2 + 1./mu) / ((dim + 2)**2 + mu))
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # History for adaptive step size
            # We'll use the standard CMA-ES update rules.
            
            local_best_f = np.inf
            local_best_x = x0.copy()
            
            # Initial evaluation
            f0 = func(x0)
            if f0 < local_best_f:
                local_best_f = f0
                local_best_x = x0.copy()
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            calls_used = 1
            
            # Main loop
            while calls_used < budget_for_run:
                if time.time() - start_time > max_time:
                    return local_best_x, local_best_f
                
                # Generate lambda samples
                # y = mean + sigma * B * z
                Z = base_rng.standard_normal((lam, dim))
                # Apply Cholesky
                # Y = Z @ B.T
                Y = Z @ B.T
                X = mean + sigma * Y
                
                # Clip to bounds
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(x) for x in X])
                calls_used += lam
                
                # Update global best
                idx_min = np.argmin(F)
                if F[idx_min] < local_best_f:
                    local_best_f = F[idx_min]
                    local_best_x = X[idx_min].copy()
                    if F[idx_min] < best_f:
                        best_f = F[idx_min]
                        best_x = X[idx_min].copy()
                
                # Sort by fitness
                idx_sorted = np.argsort(F)
                X_sorted = X[idx_sorted]
                F_sorted = F[idx_sorted]
                
                # Top mu individuals
                X_mu = X_sorted[:mu]
                F_mu = F_sorted[:mu]
                
                # Weights
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mueff = 1 / np.sum(w**2)
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(w[:, None] * X_mu, axis=0)
                
                # Update step size (sigma)
                # Evolution path p_sigma
                # norm(B^{-1} (mean - old_mean) / sigma)
                # B^{-1} = (B.T)^{-1}
                # We have mean - old_mean = sigma * B * z_bar
                # So B^{-1} (mean - old_mean) / sigma = z_bar
                # z_bar = sum(w * Z_sorted[:mu])
                Z_mu = Z[idx_sorted][:mu]
                z_bar = np.sum(w[:, None] * Z_mu, axis=0)
                
                # c_sigma
                c_sigma = (mueff + 2) / (dim + mueff + 5)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_bar / np.linalg.norm(B)
                
                # h_sigma
                # chiN = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**((calls_used + 1) / (dim * c_sigma))) < (1.4 + 2/(dim+1)) * chiN else 0
                
                # Update sigma
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                # Actually, standard is: sigma = sigma * exp(c_sigma * (||p_sigma||/chiN - 1))
                # Let's stick to the simpler form often used in implementations:
                # sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                # But c_sigma is the learning rate for p_sigma. The rate for sigma is usually c_sigma too in some variants, or derived.
                # Standard CMA-ES: sigma = sigma * exp(c_sigma * (||p_sigma||/chiN - 1))
                # Let's use that.
                
                # Update covariance matrix C
                # Evolution path p_c
                c_c = (4 + mueff/dim) / (dim + 4 + 3*mueff/dim)
                c_c = min(1, c_c) # Ensure <= 1? No, standard formula.
                # Actually c_c = (4 + mueff/dim) / (dim + 4 + 3*mueff/dim)
                
                # h_c
                chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                h_c = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**((calls_used + 1) * c_c)) < (1.4 + 2/(dim+1)) * chiN else 0
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * (mean - old_mean) / sigma
                
                # Update C
                # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1-h_c) c_c (2-c_c) C) + cmu * sum(w_i (x_i - old_mean)(x_i - old_mean)^T)
                # Note: The term (1-h_c) c_c (2-c_c) C is added if h_c=0.
                
                # Compute the rank-1 update part
                delta_h = (1 - h_c) * c_c * (2 - c_c)
                # C_new = (1 - c1 - cmu - delta_h) * C + c1 * p_c p_c^T + delta_h * C + cmu * sum(...)
                # C_new = (1 - c1 - cmu) * C + c1 * p_c p_c^T + delta_h * C + cmu * sum(...)
                # C_new = (1 - c1 - cmu + delta_h) * C + c1 * p_c p_c^T + cmu * sum(...)
                
                # Wait, the standard update is:
                # C = (1 - c1 - cmu) C + c1 (p_c p_c^T + (1-h_c) c_c (2-c_c) C) + cmu sum(...)
                # C = (1 - c1 - cmu + (1-h_c) c_c (2-c_c)) C + c1 p_c p_c^T + cmu sum(...)
                
                factor = 1 - c1 - cmu + (1 - h_c) * c_c * (2 - c_c)
                
                # Compute sum of outer products
                # D = X_mu - old_mean
                D = X_mu - old_mean
                # sum(w_i D_i D_i^T)
                # This is (D * w[:, None]).T @ D
                rank_mu = (D * w[:, None]).T @ D
                
                C = factor * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update Cholesky decomposition
                # We need B such that C = B B^T
                # We can use np.linalg.cholesky, but it's slow.
                # For dim=5, it's fast enough.
                try:
                    B = np.linalg.cholesky(C)
                except:
                    # If not positive definite, regularize
                    C = C + 1e-8 * np.eye(dim)
                    try:
                        B = np.linalg.cholesky(C)
                    except:
                        B = np.eye(dim)
                        sigma = 1.0
                        break # Restart if matrix becomes invalid
            
            return local_best_x, local_best_f

        # Multi-start strategy
        # We have `budget` calls total.
        # Let's split into restarts.
        # Each restart gets a portion of the budget.
        # Let's try 5 restarts, each with budget/5 calls.
        # But we must reserve some calls for the initial evaluation of the restart point.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for i in range(n_restarts):
            if time.time() - start_time > max_time:
                break
            
            # Generate a random starting point
            # Uniform in [-5, 5]
            x0 = base_rng.uniform(-5, 5, dim)
            
            # Run CMA-ES
            run_cmaes(x0, budget_per_restart)
            
            # Check if we should stop early if we found a very good solution?
            # Hard to know f_opt.
            # If best_f is very low, maybe stop.
            # But we don't know the scale.
            # Let's just run all restarts.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
