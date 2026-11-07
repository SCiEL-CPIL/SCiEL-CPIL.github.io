"""CMA-ES based Optimizer for BBOB functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy:
    1. Initialize a population of N=10 candidates.
    2. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) logic manually 
       to avoid scipy dependency overhead and allow precise budget control.
       - CMA-ES is robust for both smooth (Sphere, Rosenbrock) and multimodal 
         (Rastrigin, Gallagher) functions.
       - It adapts step size and covariance, allowing it to zoom in on minima 
         while maintaining diversity.
    3. We run the CMA-ES loop until the budget (300 calls) is exhausted or time limit is hit.
    
    Why this differs from the current program:
    - Current: ~250 calls on a fixed grid (wasteful for convergence) + ~50 calls L-BFGS-B.
      The grid is fixed and doesn't adapt. L-BFGS-B is local and can get stuck if the 
      grid point isn't in the right basin.
    - New: All 300 calls are used by the evolutionary process. The population adapts 
      its search region. Early calls explore, later calls exploit. This is generally 
      superior for black-box optimization with limited budgets on BBOB functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin
        
        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
                calls += 1
                all_points.append(x.copy())
                all_fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                calls += 1
                return np.inf

        # CMA-ES Parameters
        # Population size lambda
        lam = max(10, 4 + int(3 * np.log(self.dim)))
        # Initial mean: center of the box
        mean = np.zeros(self.dim)
        # Initial sigma (step size): 2.0 (covers ~40% of the range [-5,5] in 1 sigma)
        sigma = 2.0
        # Covariance matrix
        C = np.eye(self.dim)
        # Inverse square root of C (approx)
        C_inv_sqrt = np.eye(self.dim)
        
        # Weights for selection
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / (self.dim**1.8 + 10)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (self.dim**2 + 18))
        
        # State variables
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Pre-compute random numbers for reproducibility/efficiency? 
        # No, we generate on the fly.
        
        # Initial population
        # We will evaluate the initial population first.
        # To save time, we generate the population, evaluate, then update.
        
        # Note: CMA-ES usually generates new points each iteration.
        # We need to be careful with the budget. 
        # Each iteration consumes `lam` calls.
        # Total iterations possible: 300 / lam.
        # If lam=10, 30 iterations.
        
        # Let's implement the core loop.
        
        while calls + lam <= self.budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lam):
                # Standard normal
                z = np.random.standard_normal(self.dim)
                # Transform using C_inv_sqrt and sigma
                # x = mean + sigma * C_inv_sqrt @ z
                # Since we maintain C, we need C_inv_sqrt. 
                # For simplicity and speed in 5D, we can use Cholesky or just 
                # approximate C_inv_sqrt if we don't update it perfectly.
                # However, for a robust implementation, we should update C_inv_sqrt.
                # Given the low dimension (5), we can compute Cholesky each step if needed, 
                # or use the fact that C is close to identity initially.
                # Let's stick to a simpler update for C_inv_sqrt to avoid heavy linear algebra overhead 
                # unless necessary. For dim=5, np.linalg.cholesky is fast.
                
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T # L is lower, so L^-1 is lower, (L^-1)^T is upper? 
                    # Actually, C = L @ L.T. We want y such that y @ y.T = C^-1.
                    # C^-1 = L^-T @ L^-1.
                    # So C_inv_sqrt = L^-1.
                    # Let's verify: C_inv_sqrt @ C_inv_sqrt.T = L^-1 @ (L^-1).T = L^-1 @ L^-T = C^-1. Correct.
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    C_inv_sqrt = np.eye(self.dim)

                x = mean + sigma * (C_inv_sqrt @ z)
                population.append(x)
            
            # Evaluate population
            fvals = np.array([safe_call(x) for x in population])
            
            # Sort by fitness
            order = np.argsort(fvals)
            # Take top mu
            selected_indices = order[:mu]
            selected_points = np.array([population[i] for i in selected_indices])
            selected_weights = weights
            
            # Update mean
            mean = np.dot(selected_weights, selected_points)
            
            # Update sigma (CSA - Cumulative Step Adaptation)
            # h_sigma check is complex, we'll skip the h_sigma heuristic for simplicity 
            # and just update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean - mean_prev) / sigma)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (mean - mean_prev) / sigma)
            
            # Update C
            # Delta_h = 1 if ||p_sigma||^2 / E||N(0,I)||^2 < (1 + 2/sqrt(mu_eff)) * (1 - c_sigma) else 0
            # This is an approximation. We'll use a simplified version.
            h_sigma = 1.0 # Assuming no saturation for now, or compute properly
            # Proper h_sigma:
            # threshold = (1 + 2/sqrt(mu_eff)) * (1 - c_sigma)
            # norm_p_sigma_sq = np.dot(p_sigma, p_sigma)
            # E_norm_sq = self.dim # approx
            # if norm_p_sigma_sq / E_norm_sq < threshold: h_sigma = 1 else 0
            
            # Let's compute h_sigma properly
            threshold = (1 + 2 / np.sqrt(mu_eff)) * (1 - c_sigma)
            norm_p_sigma_sq = np.dot(p_sigma, p_sigma)
            # E[||N(0,I)||^2] = dim
            if norm_p_sigma_sq / self.dim < threshold:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Rank one update
            y = C_inv_sqrt @ (mean - mean_prev) / sigma
            rank_one = np.outer(y, y)
            
            # Rank mu update
            rank_mu = np.dot(selected_weights[:, np.newaxis], selected_points - mean_prev)
            rank_mu = np.dot(rank_mu, rank_mu.T) / sigma**2
            
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C + \
                c_1 * np.outer(p_c, p_c) + \
                c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))
            
            # Save previous mean for next iteration
            mean_prev = mean.copy()
            
            # Check time
            if time.time() > deadline:
                break

        # If we have remaining budget, use it for a local polish on the best point found
        # This combines the robustness of CMA-ES with the fast convergence of L-BFGS-B
        if calls < self.budget and time.time() < deadline:
            x0 = best_x if best_x is not None else np.zeros(self.dim)
            
            # Simple Nelder-Mead or L-BFGS-B polish
            # We'll use a simple gradient-free local search to avoid scipy import if not needed, 
            # but scipy is available. Let's use scipy.optimize.minimize for the polish.
            from scipy.optimize import minimize
            
            eval_count_in_minimize = 0
            def objective(x):
                nonlocal eval_count_in_minimize
                if calls + eval_count_in_minimize >= self.budget:
                    return np.inf
                if time.time() > deadline:
                    return np.inf
                eval_count_in_minimize += 1
                return safe_call(x)
                
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
