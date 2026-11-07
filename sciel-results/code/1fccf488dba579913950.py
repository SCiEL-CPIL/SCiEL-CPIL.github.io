import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 285.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        def halton_sequence(i, b):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r

        def halton_point(index, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = np.zeros(dim)
            for d in range(dim):
                point[d] = halton_sequence(index + d + 1, primes[d])
            return point

        # Phase 1: Initial Global Search
        # Focus on [-4, 4]^5 where minimizer is known to lie, with some exploration
        n_global = 50
        
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            if i < 35:
                # Halton in [-4, 4]^5 (where minimizer lies)
                x_unit = halton_point(i + 1, dim)
                x = x_unit * 8.0 - 4.0
            elif i < 45:
                # Random in [-4, 4]^5
                x = rng.uniform(-4.0, 4.0, dim)
            else:
                # Random in [-5, 5]^5 for broader coverage
                x = rng.uniform(-5.0, 5.0, dim)
            
            try:
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if best_x is None:
            return np.zeros(dim), best_f

        # Phase 2: Hierarchical Multi-Scale Stochastic Tunneling with gradient polish
        sigma = 1.5
        min_sigma = 1e-7
        max_sigma = 2.5
        
        stagnation_count = 0
        consecutive_rejections = 0
        tunnel_threshold = 35
        tunnel_sigma = 0.6
        polish_interval = 30  # polish every N calls
        calls_since_polish = 0

        while calls_made < total_budget and time.time() < deadline:
            # Periodic gradient-based polish
            if calls_since_polish >= polish_interval and calls_made < total_budget - 10:
                # Simple gradient estimation and descent
                eps = 1e-4
                for _ in range(8):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    grad = np.zeros(dim)
                    for d in range(dim):
                        x_plus = best_x.copy()
                        x_plus[d] += eps
                        x_minus = best_x.copy()
                        x_minus[d] -= eps
                        try:
                            fp = wrapper(x_plus)
                            fm = wrapper(x_minus)
                        except TimeoutError:
                            break
                        grad[d] = (fp - fm) / (2 * eps)
                    if np.linalg.norm(grad) < 1e-6:
                        break
                    # Adaptive step size based on gradient magnitude
                    lr = min(0.1 / (np.linalg.norm(grad) + 1e-8), 1.0)
                    x_new = best_x - lr * grad
                    x_new = np.clip(x_new, -5.0, 5.0)
                    try:
                        f_new = wrapper(x_new)
                    except TimeoutError:
                        break
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                    else:
                        # Try smaller step
                        lr = lr * 0.5
                        x_new = best_x + lr * grad
                        x_new = np.clip(x_new, -5.0, 5.0)
                        try:
                            f_new = wrapper(x_new)
                        except TimeoutError:
                            break
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                calls_since_polish = 0
                # Reset sigma to a small value after polish
                sigma = max(sigma * 0.5, min_sigma)

            # Generate trial point
            x_trial = best_x + sigma * rng.standard_normal(dim)
            
            try:
                fx_trial = wrapper(x_trial)
            except TimeoutError:
                break
            
            calls_since_polish += 1
            
            if fx_trial < best_f:
                # Improvement
                best_f = fx_trial
                best_x = x_trial.copy()
                stagnation_count = 0
                consecutive_rejections = 0
                # Expand sigma slightly to explore around new best
                sigma = min(sigma * 1.15, max_sigma)
            else:
                # No improvement
                consecutive_rejections += 1
                
                # Adaptive shrinking with asymmetric thresholds
                if consecutive_rejections > 4:
                    sigma = max(sigma * 0.75, min_sigma)
                    consecutive_rejections = 0
                    
                # Stagnation detection for tunneling
                stagnation_count += 1
                
                # Tunneling mechanism
                if stagnation_count > tunnel_threshold and sigma > 0.15:
                    sigma = tunnel_sigma
                    stagnation_count = 0

        return best_x, best_f
