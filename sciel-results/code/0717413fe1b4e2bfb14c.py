import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Exploration
        # Use a small number of random points to locate the region
        np.random.seed(42)
        coarse_budget = 10
        if budget <= 50:
            coarse_budget = budget // 5
            
        coarse_points = []
        for _ in range(coarse_budget):
            if time.time() - start_time > time_limit or calls >= budget:
                break
            x = np.random.uniform(-5, 5, n)
            fx = evaluate(x)
            coarse_points.append((fx, x))

        if not coarse_points:
            # Fallback if no calls made
            x = np.zeros(n)
            evaluate(x)
            coarse_points.append((best_f, x))

        # Sort by fitness
        coarse_points.sort(key=lambda item: item[0])
        
        # Phase 2: Nelder-Mead Local Search
        # We will run Nelder-Mead starting from the best point found in Phase 1
        
        # Parameters for Nelder-Mead
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around the best point
        # We use the best point and its neighbors
        # Step size for initial simplex
        delta = 1.0 
        
        x_start = coarse_points[0][1].copy()
        
        # Create initial simplex
        simplex = [(coarse_points[0][0], x_start.copy())]
        for i in range(n):
            x_neighbor = x_start.copy()
            x_neighbor[i] += delta
            x_neighbor = np.clip(x_neighbor, -5, 5)
            f_neighbor = evaluate(x_neighbor)
            simplex.append((f_neighbor, x_neighbor))
        
        # Ensure we have n+1 points
        if len(simplex) < n + 1:
             # Fill with random if evaluation failed or budget hit
             while len(simplex) < n + 1 and calls < budget:
                 x_rand = x_start + np.random.uniform(-1, 1, n)
                 x_rand = np.clip(x_rand, -5, 5)
                 f_rand = evaluate(x_rand)
                 simplex.append((f_rand, x_rand))
        
        # Nelder-Mead Main Loop
        while calls < budget and time.time() - start_time < time_limit:
            # Sort simplex by fitness
            simplex.sort(key=lambda item: item[0])
            
            f_worst = simplex[-1][0]
            x_worst = simplex[-1][1]
            f_second_worst = simplex[-2][0]
            x_second_worst = simplex[-2][1]
            f_best = simplex[0][0]
            x_best = simplex[0][1]
            
            # Check convergence
            if f_worst - f_best < 1e-10:
                break
            
            # Centroid of all but worst
            centroid = np.mean([x for _, x in simplex[:-1]], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < f_second_worst:
                # Expansion
                if f_r < f_best:
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = (f_e, x_e)
                    else:
                        simplex[-1] = (f_r, x_r)
                else:
                    simplex[-1] = (f_r, x_r)
            else:
                # Contraction
                if f_r < f_worst:
                    # Outer contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = (f_c, x_c)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][1] - x_best)
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = evaluate(x_shrink)
                            simplex[i] = (f_shrink, x_shrink)
                else:
                    # Inner contraction
                    x_c = centroid - rho * (centroid - x_worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = (f_c, x_c)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][1] - x_best)
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = evaluate(x_shrink)
                            simplex[i] = (f_shrink, x_shrink)

        return best_x, best_f
