import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global exploration with diverse starting points
        # The minimizer is in [-4, 4]^5, so sample from that region
        # Use a mix of structured and random points for better coverage
        
        starts = []
        # Origin (common for unrotated Rosenbrock)
        starts.append(np.zeros(self.dim))
        # Structured grid points at ±1, ±2, ±3 in each dimension (subset)
        # Too many for full grid in 5D, so use sparse sampling
        for _ in range(8):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
        # Also try points near the boundary of [-4,4]
        for _ in range(2):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
        
        # Evaluate all starting points
        for x0 in starts:
            if call_count >= self.budget - 50:
                break
            tracked_func(x0)
        
        # Phase 2: Local optimization from the best starting point
        # L-BFGS-B is efficient on smooth functions like Rosenbrock
        # Give it enough budget to converge (typically 50-150 calls)
        
        if time_ok() and call_count < self.budget - 20:
            remaining = self.budget - call_count - 10
            try:
                minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfun': remaining,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If we still have budget, try a second local optimization
        # from a different starting point (the second-best from Phase 1)
        if time_ok() and call_count < self.budget - 10:
            # Find the second-best starting point
            # We need to track which points were evaluated and their values
            # For simplicity, try a few more random local optimizations
            remaining = self.budget - call_count - 5
            if remaining > 20:
                # Try from a random point in the region
                x_try = rng.uniform(-3.0, 3.0, self.dim)
                tracked_func(x_try)
                if time_ok() and call_count < self.budget - 5:
                    remaining2 = self.budget - call_count - 5
                    if remaining2 > 10:
                        try:
                            minimize(
                                tracked_func,
                                x_try,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={
                                    'maxiter': 200,
                                    'maxfun': remaining2,
                                    'ftol': 1e-14,
                                    'gtol': 1e-10
                                }
                            )
                        except Exception:
                            pass
        
        # Phase 4: Final high-precision polish
        if time_ok() and call_count < self.budget - 3:
            remaining = self.budget - call_count - 3
            if remaining > 5:
                try:
                    minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining,
                            'ftol': 1e-16,
                            'gtol': 1e-14
                        }
                    )
                except Exception:
                    pass
        
        return best_x, best_fx
