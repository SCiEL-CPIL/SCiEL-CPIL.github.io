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
        deadline = start_time + 280.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Dense initial sampling for early progress (AOCC rewards early improvement)
        n_init = 40
        init_points = rng.uniform(-5, 5, size=(n_init, dim))
        
        for x in init_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: Adaptive (1+1)-ES with success-rate-based sigma adaptation
        # Structural change: single adaptive sigma instead of fixed hierarchical levels
        # The 1/5 success rule adapts sigma based on improvement rate,
        # which is more efficient than fixed step schedules for unknown landscapes.
        
        sigma = 1.0  # Initial step size
        window_size = 10  # Window for success rate calculation
        success_count = 0
        window_calls = 0
        stagnation_counter = 0
        stagnation_limit = 25
        restart_count = 0
        max_restarts = 8
        
        while calls_made < total_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.normal(0, 1, dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            candidate = best_x + direction * sigma
            candidate = np.clip(candidate, -5, 5)
            
            try:
                fc = wrapper(candidate)
            except TimeoutError:
                break
            
            improved = fc < best_f
            
            if improved:
                best_f = fc
                best_x = candidate.copy()
                success_count += 1
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            window_calls += 1
            
            # Adapt sigma every window_size calls using 1/5 success rule
            if window_calls >= window_size:
                success_rate = success_count / window_calls
                if success_rate > 0.2:
                    # Too many successes: increase sigma
                    sigma *= 1.3
                elif success_rate < 0.15:
                    # Too few successes: decrease sigma
                    sigma *= 0.7
                # else: keep sigma
                
                success_count = 0
                window_calls = 0
            
            # Stagnation detection: if no improvement for many steps, try restart
            if stagnation_counter >= stagnation_limit:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Multi-scale restart: try several different step sizes
                best_restart_f = best_f
                best_restart_x = best_x.copy()
                
                for rsigma in [sigma * 3, sigma, sigma * 0.3]:
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    rdir = rng.normal(0, 1, dim)
                    rnorm = np.linalg.norm(rdir)
                    if rnorm > 1e-10:
                        rdir /= rnorm
                    rx = np.clip(best_x + rdir * rsigma, -5, 5)
                    try:
                        fr = wrapper(rx)
                    except TimeoutError:
                        break
                    if fr < best_restart_f:
                        best_restart_f = fr
                        best_restart_x = rx.copy()
                
                if best_restart_f < best_f:
                    best_f = best_restart_f
                    best_x = best_restart_x.copy()
                
                stagnation_counter = 0
                # Reset sigma to a reasonable value
                sigma = max(0.01, min(2.0, sigma * 0.5))
        
        return best_x, best_f
