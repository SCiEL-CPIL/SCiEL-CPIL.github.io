import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Lunacek bi-Rastrigin (f24).
    Combines (1+1)-ES with adaptive step size and dynamic restarts to handle
    the deceptive multimodal landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: Multiple (1+1)-ES runs with restarts.
        # Lunacek has a global min and a local min.
        # The global min is often deeper but harder to reach (narrow valley).
        # The local min is easier to reach (wide basin) but shallower.
        
        # We use a restart policy:
        # 1. Start at origin (0,0,0,0,0) which is usually close to the global min in BBOB f24.
        # 2. If no improvement for a while, or if we seem stuck, restart from a perturbed best point or random point.
        
        # We divide the budget into segments.
        # If a segment fails to improve the global best, we restart.
        
        num_restarts = 10  # Max restarts
        # Dynamic budget allocation:
        # First restart gets a larger chunk to find the global valley.
        # Subsequent restarts get smaller chunks to polish or escape local mins.
        
        # Let's use a geometric decay for restart budgets
        # Total budget = 500
        # 50, 40, 30, 20, 15, 10, 10, 10, 10, 15 (last one takes remainder)
        
        budgets = []
        remaining = total_budget
        decay = 0.8
        current_budget = int(total_budget * 0.25) # 25% for first
        
        for i in range(num_restarts):
            if remaining <= 0:
                break
            b = min(current_budget, remaining)
            if b < 5:
                # If remaining is small, just give it to the last one
                if i == num_restarts - 1:
                    b = remaining
                else:
                    continue
            budgets.append(b)
            remaining -= b
            current_budget = int(current_budget * decay)
        
        # Ensure we use all budget
        if remaining > 0 and budgets:
            budgets[-1] += remaining
        elif remaining > 0:
            budgets.append(remaining)

        for restart_idx, b in enumerate(budgets):
            if calls_made >= total_budget or time.time() > deadline:
                break

            # Initialization for this restart
            if restart_idx == 0:
                # Start at origin, standard for BBOB f24
                x = np.zeros(n)
            elif restart_idx == 1:
                # Try a point in the other basin (local min area)
                # In BBOB f24, the local min is often shifted.
                # A good guess is a point with larger norm.
                x = rng.uniform(-4, 4, n)
            else:
                # Perturb the best solution found so far
                if best_x is not None:
                    # Scale of perturbation: large enough to escape, small enough to stay in bounds
                    # If we are stuck in a local min, we might need a larger jump
                    sigma_init = 2.0
                    x = np.clip(best_x + rng.standard_normal(n) * sigma_init, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)

            # Evaluate initial point
            fx = safe_call(x)
            if np.isinf(fx):
                break
            
            # (1+1)-ES parameters
            sigma = 1.0  # Initial step size
            window = 10  # Window for 1/5 rule
            successes = 0
            window_count = 0
            no_improvement_count = 0
            max_no_improvement = 30  # Restart if no improvement for 30 steps
            
            # Main loop for this restart
            while calls_made < total_budget and time.time() < deadline:
                # Check if we've used up this restart's budget
                # We need to track how many calls we've made in this restart
                # Let's just use a simple counter
                # Actually, it's easier to just run until budget is exhausted or no improvement
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_call(y)
                
                if np.isinf(fy):
                    break

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    successes += 1
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                window_count += 1

                # Adapt sigma using 1/5 success rule
                if window_count >= window:
                    rate = successes / window_count
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.15:
                        sigma *= 0.8
                    successes = 0
                    window_count = 0
                    sigma = np.clip(sigma, 1e-8, 5.0)

                # Stagnation check
                if no_improvement_count > max_no_improvement:
                    break

                # If sigma is very small, we might be converged
                if sigma < 1e-6:
                    break

        # Final polish: Coordinate descent on the best point found
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            # Try small steps in each coordinate
            steps = [0.1, 0.01, 0.001]
            for step in steps:
                for i in range(n):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    # Positive
                    y = x.copy()
                    y[i] = min(5.0, x[i] + step)
                    fy = safe_call(y)
                    if not np.isinf(fy) and fy < fx:
                        x = y.copy()
                        fx = fy
                    # Negative
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    y = x.copy()
                    y[i] = max(-5.0, x[i] - step)
                    fy = safe_call(y)
                    if not np.isinf(fy) and fy < fx:
                        x = y.copy()
                        fx = fy

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
