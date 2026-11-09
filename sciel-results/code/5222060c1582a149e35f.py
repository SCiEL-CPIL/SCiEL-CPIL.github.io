"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Schwefel f20 on [-5,5]^5.
    
    Strategy:
    1. Quick random sampling to find a good starting region.
    2. CMA-ES-like adaptive step-size search with restarts.
    3. L-BFGS-B refinement from the best point found.
    
    Includes a deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return None
            call_count += 1
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return None
        
        # Phase 1: Random sampling to find a good starting point
        # Use 50 random points to get a decent initial guess
        n_random = min(50, self.budget)
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            if fx is None:
                break
        
        if best_x is None:
            # Fallback: return a default point
            return np.zeros(self.dim), 0.0
        
        # Phase 2: CMA-ES-like adaptive search
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # Adaptive step size parameters
        success_ratio = 0.5  # Target success rate
        last_success = 0
        window_size = 10
        success_count = 0
        total_count = 0
        
        max_cma_iter = self.budget - call_count - 50  # Reserve some for L-BFGS-B
        max_cma_iter = max(10, min(max_cma_iter, 300))
        
        for i in range(max_cma_iter):
            if time.time() > deadline or call_count >= self.budget:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = safe_call(y)
            if fy is None:
                break
            
            # Track success
            total_count += 1
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                # Increase step size on success
                sigma *= 1.3
            else:
                # Decrease step size on failure
                sigma *= 0.85
            
            # Adaptive sigma adjustment based on success rate
            if total_count >= window_size:
                current_rate = success_count / total_count
                if current_rate > 0.7:
                    sigma *= 1.2
                elif current_rate < 0.3:
                    sigma *= 0.8
                # Reset window
                success_count = 0
                total_count = 0
            
            # If sigma gets too small, restart from best with larger sigma
            if sigma < 0.01 and i > 20:
                # Restart with new random direction
                x = best_x + 2.0 * rng.standard_normal(self.dim)
                x = np.clip(x, -5, 5)
                fx = safe_call(x)
                if fx is None:
                    break
                sigma = 1.0
                success_count = 0
                total_count = 0
        
        # Phase 3: L-BFGS-B refinement if budget and time remain
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                
                # Use best point as starting point
                x0 = best_x.copy()
                
                # Limit number of function evaluations for L-BFGS-B
                maxiter = min(remaining, 100)
                
                res = minimize(
                    func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': maxiter, 'maxfun': min(remaining, 200)}
                )
                
                if res.success and res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
                    
            except Exception:
                pass  # Ignore errors from L-BFGS-B
        
        return best_x, best_f
# EVOLVE-BLOCK-END
