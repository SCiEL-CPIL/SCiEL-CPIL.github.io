"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    The Bent Cigar function (BBOB f12) is highly anisotropic. CMA-ES is a 
    state-of-the-art black-box optimizer that adapts the covariance matrix 
    of the search distribution to the local landscape geometry. This allows 
    it to align with the narrow valley of the Bent Cigar function much more 
    effectively than methods that do not explicitly model correlations between 
    variables, such as Nelder-Mead or simple coordinate descent.
    
    Key features:
    1. Population-based search provides robustness against local minima.
    2. Adaptive step size and covariance matrix allow rapid convergence 
       along the principal axes of the function's Hessian.
    3. No derivative information required.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # CMA-ES Parameters
        # For dim=5, a population size of 10 is a good balance between 
        # exploration and exploitation.
        lam = 10
        mu = lam // 2  # Weight top mu individuals
        
        # Weights for the selected individuals
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Step size control
        cs = (mueff + 1.5) / (dim + mueff + 3)
        # Covariance matrix control
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # Damping for step size
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Initial state
        # Start from the center of the box
        xmean = np.zeros(dim)
        sigma = 2.0  # Initial step size, box is [-5,5] so 2.0 covers a good portion
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky factor of C
        C_sqrt_inv = np.eye(dim) # We will maintain C and its inverse square root
        # Actually, it's easier to maintain C and compute Cholesky each step for sampling,
        # or maintain C_inv_sqrt. For dim=5, computing Cholesky is cheap.
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluations
        # We need to evaluate the initial population to start the loop.
        # However, CMA-ES usually generates the population, evaluates, then updates.
        # We will do one initial evaluation of xmean to have a baseline.
        
        try:
            f_mean = wrapped_func(xmean)
        except (TimeoutError, ValueError):
            return np.zeros(dim), 0.0

        # Main loop
        # Each generation consumes 'lam' evaluations.
        # Total budget is 500.
        # Generations = 500 // 10 = 50.
        
        while self.calls < budget and time.time() < deadline:
            # 1. Sample population
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Sample standard normal vectors
            zs = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=lam)
            
            # Transform to search points
            # x_k = xmean + sigma * (L @ z_k)
            # Note: L @ z is equivalent to sqrt(C) @ z if L is the Cholesky factor such that C = L L^T
            # Standard CMA-ES uses B sigma z where B B^T = C.
            # If C = L L^T, then x = xmean + sigma * L @ z
            
            xs = xmean + sigma * (zs @ L.T)
            
            # Evaluate population
            fs = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline or self.calls >= budget:
                    break
                try:
                    fs[i] = wrapped_func(xs[i])
                except (TimeoutError, ValueError):
                    # If we run out of budget/time mid-generation, stop
                    # Fill remaining with inf to allow sorting if needed, but we break loop
                    fs[i] = float('inf')
            
            # Check if we broke out early due to limits
            if time.time() > deadline or self.calls >= budget:
                break

            # 2. Sort by function value
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # 3. Update mean
            # New mean is the weighted average of the top mu individuals
            xmean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # 4. Update step size (sigma)
            # Compute norm of the old C_sqrt_inv @ p_sigma
            # We need C_inv_sqrt. Let's compute it from L.
            # C = L L^T. C_inv = (L L^T)^-1 = L^-T L^-1.
            # C_inv_sqrt = L^-T.
            # Solving L v = I gives L^-1. So C_inv_sqrt = (L^-1).T
            try:
                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            # p_sigma = (1 - cs) p_sigma + sqrt(cs (2 - cs) mueff) C_inv_sqrt (xmean_new - xmean) / sigma
            diff_mean = (xmean_new - xmean) / sigma
            norm_ps = np.linalg.norm(C_inv_sqrt @ diff_mean)
            hat_cs = np.sqrt(cs * (2 - cs) * mueff)
            p_sigma = (1 - cs) * p_sigma + hat_cs * C_inv_sqrt @ diff_mean
            
            # Update sigma
            # log-normal adaptation
            sigma = sigma * np.exp(cs / damps * (norm_ps / np.linalg.norm(np.ones(dim)) - 1))
            
            # 5. Update covariance matrix
            # Rank-one update
            # y = (xmean_new - xmean) / sigma
            y = (xmean_new - xmean) / sigma
            h_sigma = (norm_ps < (1.4 + 2/(dim+1)) * np.sqrt((1 - (1-cs)**2) / (1 - (1-cs)**(self.calls)))) # Approximation for h_sigma
            # A more standard check for h_sigma involves the history of p_sigma, 
            # but for simplicity and robustness in a short run, we can use a simplified version 
            # or just perform the update. The standard h_sigma is:
            # h_sigma = 1 if norm_ps < (1 + 2/(dim+1)) * sqrt(1 - (1-cs)^2) else 0
            # Let's use the simpler threshold
            if norm_ps < (1.4 + 2.0/(dim+1.0)) * np.sqrt(1.0 - (1.0-cs)**2):
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mueff) * y
            
            # C = (1 + c1 * (2 - c1) - c1) C + c1 (p_c p_c^T + (1-h_sigma) c1 (2-c1) C)
            # Wait, the standard update is:
            # C = (1 - c1 - cmu) C + c1 (p_c p_c^T + (1-h_sigma) c1 (2-c1) C) + cmu sum w_i (y_i y_i^T)
            
            # Let's stick to the standard formulation:
            # C = (1 - c1 - cmu) * C
            C = (1 - c1 - cmu) * C
            
            # Add rank-one term from p_c
            C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c1 * (2 - c1) * np.eye(dim))
            
            # Add rank-mu term from selected individuals
            # y_i = (x_i - xmean_old) / sigma
            # Note: We use the OLD xmean for the difference in the rank-mu update?
            # Yes, usually the difference is taken with respect to the mean BEFORE the update.
            for i in range(mu):
                y_i = (xs_sorted[i] - xmean) / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            
            # Update xmean
            xmean = xmean_new
            
            # Convergence check
            # If the spread of function values is small, or if sigma is very small
            if np.max(fs_sorted) - np.min(fs_sorted) < 1e-12:
                break
            if sigma < 1e-12:
                break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
