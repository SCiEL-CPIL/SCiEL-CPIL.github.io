import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Random exploration to find promising basins
        # Rastrigin has many local minima. Random starts help locate the global basin.
        n_random = 10
        for _ in range(n_random):
            if not time_ok() or calls_made >= total_budget:
                break
            x = rng.uniform(-5.0, 5.0, n)
            do_call(x)

        # Phase 2: Parallel Nelder-Mead Ensemble
        # Initialize 5 simplexes at diverse locations: center, 4 corners, and best found.
        # This allows simultaneous exploration of different basins.
        
        def nelder_mead(start_x, budget_limit, max_iter=100):
            """
            Runs Nelder-Mead optimization from start_x until budget_limit or max_iter.
            Returns best_x, best_f found within this simplex.
            """
            nonlocal calls_made, best_f, best_x
            
            # Initialize simplex
            simplex = [start_x.copy()]
            f_simplex = [do_call(start_x)]
            
            # If start was out of budget, return
            if calls_made >= total_budget:
                return best_x, best_f

            # Expand simplex
            for d in range(n):
                if calls_made >= total_budget or not time_ok():
                    break
                x_new = start_x.copy()
                # Step size: 0.5 if not zero, else 0.001
                step = 0.5 if abs(start_x[d]) > 1e-8 else 0.001
                x_new[d] += step
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = do_call(x_new)
                simplex.append(x_new)
                f_simplex.append(f_new)
            
            if len(simplex) < n + 1:
                return best_x, best_f

            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink

            for _ in range(max_iter):
                if calls_made >= total_budget or not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_simplex[-1]
                f_best = f_simplex[0]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = do_call(x_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = do_call(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = do_call(x_c)
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_simplex[i] = do_call(simplex[i])
                        
                        # Check if we should stop early if budget is tight
                        if calls_made >= total_budget or not time_ok():
                            break

        # Define starting points for the ensemble
        # 1. Best found so far
        # 2. Center (0,0,0,0,0)
        # 3. Corner (4,4,4,4,4)
        # 4. Corner (-4,-4,-4,-4,-4)
        # 5. Mixed corner (4,-4,4,-4,4)
        
        start_points = [
            best_x.copy(),
            np.zeros(n),
            np.array([4.0] * n),
            np.array([-4.0] * n),
            np.array([4.0 if i % 2 == 0 else -4.0 for i in range(n)])
        ]

        # Run Nelder-Mead for each start point
        # We distribute the remaining budget roughly equally, but let them run until budget is exhausted.
        # Since they run sequentially here, we just loop.
        
        for sp in start_points:
            if calls_made >= total_budget or not time_ok():
                break
            nelder_mead(sp, total_budget)

        return best_x, best_f
