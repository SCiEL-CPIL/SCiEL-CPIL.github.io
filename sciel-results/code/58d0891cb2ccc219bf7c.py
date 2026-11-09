import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = self.dim

        # Phase 1: (1+1)-ES with adaptive sigma and restarts
        # Use ~70% of budget for global exploration
        global_budget = int(self.budget * 0.70)

        # Start at origin
        x = np.zeros(n)
        try:
            safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f

        sigma = 1.0
        stall_count = 0
        stall_limit = 50
        best_local = best_f
        best_local_x = best_x.copy() if best_x is not None else x.copy()

        while calls < global_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(n)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5.0, 5.0)

            try:
                f_candidate = safe_func(candidate)
            except (TimeoutError, StopIteration):
                break

            if f_candidate < best_local - 1e-12:
                # Success: move to candidate, grow sigma
                best_local = f_candidate
                best_local_x = candidate.copy()
                x = candidate.copy()
                sigma *= 1.15
                stall_count = 0
            else:
                # Failure: stay, shrink sigma
                sigma *= 0.92
                stall_count += 1

                # If stalled too long, restart from a diverse point
                if stall_count >= stall_limit:
                    # Perturb the best point found so far with moderate noise
                    if best_local_x is not None:
                        x = best_local_x + rng.standard_normal(n) * 0.5
                    else:
                        x = rng.uniform(-4.0, 4.0, n)
                    x = np.clip(x, -5.0, 5.0)
                    sigma = 1.0
                    stall_count = 0
                    try:
                        fx = safe_func(x)
                        if fx < best_local:
                            best_local = fx
                            best_local_x = x.copy()
                    except (TimeoutError, StopIteration):
                        break

        # Phase 2: Multi-start Nelder-Mead with dynamic budget allocation
        remaining = self.budget - calls
        if remaining > 20 and best_x is not None and time.time() < deadline:
            # Generate diverse starting points for multi-start Nelder-Mead
            n_starts = min(10, max(3, remaining // 30))
            start_points = []
            
            # Always include the global best
            start_points.append(best_x.copy())
            
            # Add perturbed versions of the best point
            for i in range(1, n_starts):
                if i < 3:
                    # Small perturbations around best
                    pert = rng.standard_normal(n) * (0.1 + 0.05 * i)
                elif i < 6:
                    # Medium perturbations
                    pert = rng.standard_normal(n) * (0.3 + 0.1 * i)
                else:
                    # Larger perturbations to explore other basins
                    pert = rng.uniform(-1.0, 1.0, n)
                
                new_point = best_x + pert
                new_point = np.clip(new_point, -5.0, 5.0)
                start_points.append(new_point)
            
            # Allocate budget dynamically: give more to promising starts
            # First, evaluate all starting points to rank them
            start_values = []
            for sp in start_points:
                try:
                    val = safe_func(sp)
                    start_values.append(val)
                except (TimeoutError, StopIteration):
                    start_values.append(np.inf)
            
            # Rank starts by their initial value
            ranked_indices = np.argsort(start_values)
            
            # Allocate remaining budget proportionally to convergence potential
            # Give more budget to better starting points
            base_budget = max(5, (remaining - len(start_points)) // n_starts)
            
            for idx in ranked_indices:
                if calls >= self.budget or time.time() > deadline - 5:
                    break
                    
                # Better starts get more iterations
                if idx == 0:
                    max_iter = min(remaining - calls, base_budget * 2)
                elif idx == 1:
                    max_iter = min(remaining - calls, base_budget * 1.5)
                else:
                    max_iter = min(remaining - calls, base_budget)
                
                if max_iter < 3:
                    continue
                
                try:
                    result = scipy.optimize.minimize(
                        safe_func,
                        start_points[idx],
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-9,
                            'fatol': 1e-11,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

        return best_x, best_f
