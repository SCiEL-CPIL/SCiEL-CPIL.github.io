import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer combining (1+1)-ES exploration with L-BFGS-B and Powell polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Track calls to ensure we don't exceed budget
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES exploration to find a good starting point
        # Bent Cigar has a narrow valley; adaptive step sizing helps locate it
        rng = np.random.default_rng()
        es_budget = int(self.budget * 0.3)  # Use 30% of budget for exploration
        
        x = rng.uniform(-5, 5, self.dim)
        fx = wrapped_func(x)
        sigma = 2.0
        success_count = 0
        trial_count = 0
        window_size = 10  # 1/5 success rule window
        
        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            if call_count >= self.budget - 50:  # Leave budget for refinement
                break
                
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                trial_count += 1
                # Adapt sigma based on 1/5 success rule
                if trial_count >= window_size:
                    if success_count / trial_count > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    success_count = 0
                    trial_count = 0
            else:
                trial_count += 1
                if trial_count >= window_size:
                    if success_count / trial_count > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    success_count = 0
                    trial_count = 0
        
        # Update best if ES found something better
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # Phase 2: Local optimization with L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        remaining_budget = self.budget - call_count

        if remaining_budget > 20 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfev': remaining_budget, 'ftol': 1e-12, 'gtol': 1e-10, 'eps': 1e-8}
                )
            except Exception:
                pass

        # Phase 3: Final high-precision polish with Powell's method
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and time.time() < deadline:
            try:
                result2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': remaining_budget, 'ftol': 1e-14, 'xtol': 1e-12, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f
