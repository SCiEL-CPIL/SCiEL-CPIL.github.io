"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Schwefel f20 on [-5,5]^5.
    
    Strategy:
    1. Broad random sampling to find multiple good starting regions.
    2. Multi-start L-BFGS-B from top-k promising points.
    3. (1+1)-ES polish from the best result with adaptive step size.
    4. Final tight-tolerance L-BFGS-B refinement.
    
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
        
        # Phase 1: Broad random sampling to find multiple good starting points
        # Use 80 random points to better explore the landscape
        n_random = min(80, self.budget)
        candidates = []
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            if fx is None:
                break
            candidates.append((fx, x.copy()))
        
        if best_x is None:
            # Fallback: return a default point
            return np.zeros(self.dim), 0.0
        
        # Sort candidates by function value and keep top-k
        candidates.sort(key=lambda t: t[0])
        top_k = candidates[:min(5, len(candidates))]
        
        # Phase 2: Multi-start L-BFGS-B from top-k promising points
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize
                
                for fx_init, x_init in top_k:
                    if call_count >= self.budget or time.time() > deadline - 5:
                        break
                    
                    # Limit number of function evaluations for L-BFGS-B
                    maxfun_per_start = min(remaining // len(top_k), 50)
                    
                    res = minimize(
                        func,
                        x_init,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 100, 'maxfun': maxfun_per_start}
                    )
                    
                    if res.success and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                    
                    remaining = self.budget - call_count
                    if remaining < 10:
                        break
                        
            except Exception:
                pass  # Ignore errors from L-BFGS-B
        
        # Phase 3: (1+1)-ES polish from the best result
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate step size
            
            # Adaptive step size parameters
            window_size = 8
            success_count = 0
            total_count = 0
            
            max_es_iter = min(remaining - 10, 150)
            
            for i in range(max_es_iter):
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
                    sigma *= 1.25
                else:
                    # Decrease step size on failure
                    sigma *= 0.8
            
                # Adaptive sigma adjustment based on success rate
                if total_count >= window_size:
                    current_rate = success_count / total_count
                    if current_rate > 0.6:
                        sigma *= 1.15
                    elif current_rate < 0.25:
                        sigma *= 0.75
                    # Reset window
                    success_count = 0
                    total_count = 0
                
                # If sigma gets too small, try a larger jump
                if sigma < 0.05 and i > 15:
                    x = best_x + 1.5 * rng.standard_normal(self.dim)
                    x = np.clip(x, -5, 5)
                    fx = safe_call(x)
                    if fx is None:
                        break
                    sigma = 0.5
                    success_count = 0
                    total_count = 0
        
        # Phase 4: Final tight-tolerance L-BFGS-B refinement
        remaining = self.budget - call_count
        if remaining > 5 and time.time() < deadline - 2:
            try:
                from scipy.optimize import minimize
                
                res = minimize(
                    func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': min(remaining, 100), 
                             'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success and res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
                    
            except Exception:
                pass  # Ignore errors from L-BFGS-B
        
        return best_x, best_f
# EVOLVE-BLOCK-END
