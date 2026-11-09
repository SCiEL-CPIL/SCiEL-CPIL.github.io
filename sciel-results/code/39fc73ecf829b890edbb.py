"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES Optimizer for BBOB f21 (Gallagher 101 peaks).
    
    Strategy:
    1. Dense Initial Screening: Use 50 evaluations to find a good starting point 
       and establish a low baseline quickly for AOCC.
    2. CMA-ES: Run a single, continuous CMA-ES instance for the remaining budget.
       CMA-ES is chosen because it adapts the covariance matrix to the local 
       landscape structure, which is crucial for the rotated and multi-modal 
       nature of f21. Unlike DE or NM, it maintains a population that models 
       the correlation structure of the search region.
    3. Safety: Explicit time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        global_best_x = None
        global_best_f = float('inf')
        total_calls = 0
        start_time = time.time()
        max_time = 280.0  # Safety margin for 330s limit

        def can_continue():
            return total_calls < budget and (time.time() - start_time) < max_time

        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # --- Phase 1: Dense Initial Screening ---
        # Use 50 evaluations as suggested by the directive to seed the mean.
        n_scan = 50
        scan_results = []
        
        for i in range(n_scan):
            if not can_continue():
                break
            
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            
            update_best(x, fx)
            scan_results.append((fx, x))
            
            if not can_continue():
                break

        if global_best_x is None:
            x0 = np.zeros(dim)
            fx = func(x0)
            total_calls += 1
            update_best(x0, fx)
            scan_results.append((fx, x0))

        # Sort to get the best candidate
        scan_results.sort(key=lambda item: item[0])
        best_start_x = scan_results[0][1].copy()
        
        # --- Phase 2: CMA-ES ---
        
        class CMAES:
            def __init__(self, x0, sigma0, func, update_best, total_calls_ref, can_continue_fn):
                self.n = len(x0)
                self.x_mean = x0.copy()
                self.sigma = sigma0
                self.C = np.eye(self.n)
                self.pc = np.zeros(self.n)
                self.ps = np.zeros(self.n)
                
                # Parameters for n=5
                # pop_size = 4 + 3 * ln(n) = 4 + 3*ln(5) ~ 8.9 -> 9
                self.pop_size = 9 
                mu = self.pop_size // 2  # 4
                
                # Weights
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                self.weights = weights
                self.mu_eff = 1.0 / np.sum(weights**2)
                
                # Adaptation rates
                self.c_sigma = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
                self.c_c = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
                self.c_1 = 2 / ((self.n + 1.3)**2 + self.mu_eff)
                self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2)**2 + self.mu_eff))
                
                self.chi_n = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
                
                self.func = func
                self.update_best = update_best
                self.total_calls_ref = total_calls_ref
                self.can_continue = can_continue_fn

            def step(self):
                if not self.can_continue():
                    return False
                
                # 1. Sampling
                pop = np.zeros((self.pop_size, self.n))
                # Ensure C is positive definite
                try:
                    L = np.linalg.cholesky(self.C)
                except np.linalg.LinAlgError:
                    self.C = np.eye(self.n)
                    L = np.eye(self.n)
                
                for i in range(self.pop_size):
                    z = rng.standard_normal(self.n)
                    pop[i] = self.x_mean + self.sigma * (L @ z)
                
                # 2. Evaluation
                fvals = np.zeros(self.pop_size)
                for i in range(self.pop_size):
                    if not self.can_continue():
                        return False
                    fvals[i] = self.func(pop[i])
                    self.total_calls_ref[0] += 1
                    self.update_best(pop[i], fvals[i])
                
                # 3. Sorting
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]
                
                # 4. Update Mean
                # Weighted sum of the best mu points
                y_w = np.sum(self.weights[:, None] * pop_sorted[:mu], axis=0)
                
                # 5. Update Path Variables
                # Standardize y_w with current sigma and C
                # C^{-1/2} y_w
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                    y_w_std = C_inv_sqrt @ y_w
                except np.linalg.LinAlgError:
                    y_w_std = y_w
                
                self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_w_std
                h_sig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**2) / self.chi_n) < (1.4 + 2 / (self.n + 1))
                
                # pc update
                if h_sig:
                    self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w_std
                else:
                    self.pc = (1 - self.c_c) * self.pc

                # 6. Update Covariance Matrix
                # Delta H term
                delta_h = (2 - self.c_c) * h_sig - self.c_c
                C_old = self.C.copy()
                
                # Rank-mu update
                v = np.zeros((self.n, mu))
                for i in range(mu):
                    v[:, i] = C_inv_sqrt @ (pop_sorted[i] - self.x_mean) / self.sigma
                    # Note: y_w is already weighted sum of (pop - mean)/sigma? 
                    # Standard CMA-ES: y_w = sum w_i (x_i - mean). 
                    # Here we used y_w_std = C^{-1/2} y_w / sigma? 
                    # Let's stick to the standard formulation:
                    # y_w = sum w_i (x_i - x_mean)
                    # The update is C <- (1 - c_1 - c_mu) C + c_1 (pc pc^T + (delta_h - 1 + c_c) C) + c_mu sum w_i v_i v_i^T
                    # where v_i = C^{-1/2} (x_i - x_mean) / sigma.
                    # My y_w_std calculation above: y_w_std = C^{-1/2} y_w.
                    # y_w = sum w_i (x_i - x_mean).
                    # So v_i should be C^{-1/2} (x_i - x_mean) / sigma.
                    # Let's recalculate v correctly.
                    pass
                
                # Recalculate v correctly
                v = np.zeros((self.n, mu))
                for i in range(mu):
                    v[:, i] = C_inv_sqrt @ (pop_sorted[i] - self.x_mean) / self.sigma
                
                rank_mu = np.sum(self.weights[:, None, None] * v @ v.transpose(0, 1), axis=2)
                
                self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                         self.c_1 * (np.outer(self.pc, self.pc) + (delta_h - 1 + self.c_c) * C_old) + \
                         self.c_mu * rank_mu

                # 7. Update Mean
                self.x_mean = self.x_mean + y_w

                # 8. Update Sigma
                self.sigma *= np.exp(self.c_sigma / self.chi_n * (np.linalg.norm(self.ps) / self.chi_n - 1))
                
                # Clamp sigma to avoid explosion or vanishing
                self.sigma = np.clip(self.sigma, 1e-10, 10.0)

                # Check for stagnation/convergence
                if self.sigma < 1e-10:
                    return False

                return True

        # Initialize CMA-ES
        # Initial sigma: heuristic based on box size
        sigma0 = 0.3 * (5 - (-5)) / 2  # 1.5? Or maybe smaller. 
        # For f21, peaks are narrow. A moderate start is good.
        sigma0 = 1.0 
        
        cma = CMAES(best_start_x, sigma0, func, update_best, [total_calls], can_continue)
        
        while can_continue():
            success = cma.step()
            if not success:
                break
        
        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
