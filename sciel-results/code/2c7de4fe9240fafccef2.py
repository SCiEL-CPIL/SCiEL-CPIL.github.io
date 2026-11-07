import numpy as np
import time

class Optimizer:
    """Multi-start Hooke-Jeeves pattern search for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        candidates = []

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def hooke_jees(start_x):
            nonlocal eval_count
            if eval_count >= budget - 10:  # Leave some buffer
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            step_sizes = np.ones(d) * 0.5
            min_step = 1e-4
            max_iter = 200  # Limit iterations to prevent infinite loops if stuck

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                        # Continue exploring this direction if it improved
                        # (Standard HJ often resets the loop or continues, 
                        # here we continue to the next coordinate to maintain 
                        # a structured search, but we could also loop back.
                        # Sticking to the inspiration's flow: continue to next i)
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step) or not moved:
                    # If no move was made in a full cycle, we might be stuck or converged
                    # If step sizes are small, we are converged.
                    if np.all(step_sizes < min_step):
                        break
                    # If no move but steps are large, try shrinking all steps slightly?
                    # Or just break if we didn't move at all to avoid wasting budget
                    if not moved:
                        break

        # Phase 1: Global Coarse Search / Multi-Start Initialization
        rng = np.random.default_rng(42)
        
        # Generate initial candidates: Origin and random points
        init_points = [np.zeros(d)]
        for _ in range(9):
            # Sample in [-4, 4] as per problem statement minimizer location
            init_points.append(rng.uniform(-4, 4, d))
            
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
            
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement on the best starting points
        # Try to refine the top 3 points if budget allows
        top_k = min(3, len(point_scores))
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            # Refine the i-th best point
            # We re-evaluate the starting point inside hooke_jees? 
            # The eval_func inside hooke_jees will count it. 
            # To save budget, we could pass the known f, but hooke_jees 
            # in the inspiration re-evaluates. Let's modify hooke_jees 
            # to accept an initial f to save 1 call per restart?
            # Actually, the cost of 1 call is negligible compared to the search.
            hooke_jees(point_scores[i][1])

        return best_x, best_f
