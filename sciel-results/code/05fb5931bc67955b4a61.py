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
        time_limit = 280.0

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

        def time_left():
            return start_time + time_limit - time.time()

        # Phase 1: Coarse Exploration
        # Use a small number of random points to locate the region
        np.random.seed(42)
        coarse_budget = 10
        if budget <= 50:
            coarse_budget = max(1, budget // 5)
            
        coarse_points = []
        for _ in range(coarse_budget):
            if time_left() <= 0 or calls >= budget:
                break
            x = np.random.uniform(-5, 5, n)
            fx = evaluate(x)
            coarse_points.append((fx, x))

        if not coarse_points:
            x = np.zeros(n)
            evaluate(x)
            coarse_points.append((best_f, x))

        coarse_points.sort(key=lambda item: item[0])
        x_start = coarse_points[0][1].copy()

        # Phase 2: Nelder-Mead Local Search
        # We will run Nelder-Mead starting from the best point found in Phase 1
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        delta = 1.0 
        
        # Create initial simplex
        simplex = [(coarse_points[0][0], x_start.copy())]
        for i in range(n):
            x_neighbor = x_start.copy()
            x_neighbor[i] += delta
            x_neighbor = np.clip(x_neighbor, -5, 5)
            f_neighbor = evaluate(x_neighbor)
            simplex.append((f_neighbor, x_neighbor))
        
        if len(simplex) < n + 1:
             while len(simplex) < n + 1 and calls < budget and time_left() > 0:
                 x_rand = x_start + np.random.uniform(-1, 1, n)
                 x_rand = np.clip(x_rand, -5, 5)
                 f_rand = evaluate(x_rand)
                 simplex.append((f_rand, x_rand))
        
        # Nelder-Mead Main Loop
        while calls < budget and time_left() > 0:
            simplex.sort(key=lambda item: item[0])
            
            f_worst = simplex[-1][0]
            x_worst = simplex[-1][1]
            f_second_worst = simplex[-2][0]
            x_second_worst = simplex[-2][1]
            f_best = simplex[0][0]
            x_best = simplex[0][1]
            
            if f_worst - f_best < 1e-10:
                break
            
            centroid = np.mean([x for _, x in simplex[:-1]], axis=0)
            
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < f_second_worst:
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
                if f_r < f_worst:
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = (f_c, x_c)
                    else:
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][1] - x_best)
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = evaluate(x_shrink)
                            simplex[i] = (f_shrink, x_shrink)
                else:
                    x_c = centroid - rho * (centroid - x_worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < f_worst:
                        simplex[-1] = (f_c, x_c)
                    else:
                        for i in range(1, len(simplex)):
                            x_shrink = x_best + sigma * (simplex[i][1] - x_best)
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = evaluate(x_shrink)
                            simplex[i] = (f_shrink, x_shrink)
                            
            if time_left() <= 0:
                break

        # Phase 3: CMA-ES Refinement
        # Use remaining budget for a more robust global/local search
        if calls < budget and time_left() > 1.0:
            # CMA-ES parameters
            # lambda_ = 4 + floor(3 * log(dim + 1))
            lambda_ = max(4, int(4 + 3 * np.log(n + 1)))
            mu = max(1, lambda_ // 2)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 0.5) / (n + 2)
            d_sigma = 1 + 2 * max(0, (np.sqrt((mu_eff + 1) / (n + 1)) - 1) / (n + 2)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Initialize state around current best
            mean = best_x.copy()
            sigma = 0.5  # Start small since we already converged locally
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute Cholesky if possible, but we'll update C so we need to do it every iter or use eig
            # For dim=5, Cholesky is cheap.
            
            # Track calls
            while calls < budget and time_left() > 0.1:
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if calls >= budget or time_left() <= 0.1:
                        break
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    
                    z = np.random.standard_normal(n)
                    y = mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = evaluate(y)
                    offspring.append((fy, y))
                
                if len(offspring) < mu:
                    break
                    
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Selection
                selected = [y for _, y in offspring[:mu]]
                
                # Weighted mean
                m = np.zeros(n)
                for i in range(mu):
                    m += weights[i] * selected[i]
                
                # Update evolution paths
                if sigma > 1e-10:
                    diff = (m - mean) / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                    
                    # Simplified h_sigma check
                    # h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)) - 1 else 0.0
                    # Using a simpler heuristic for speed and stability in small dim
                    norm_p_sigma = np.linalg.norm(p_sigma)
                    threshold = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)) - 1.0
                    h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
                    
                    p_c = (1 - c_c) * p_c + h_sigma * c_c * diff
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(selected[i] - mean, selected[i] - mean) for i in range(mu))
                    
                    # Ensure C is positive definite
                    try:
                        np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Add small identity
                        C += 1e-6 * np.eye(n)
                        
                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
                    sigma = max(sigma, 1e-10)
                    sigma = min(sigma, 10.0)
                    
                # Update mean
                mean = m
                mean = np.clip(mean, -5, 5)
                
                # Check convergence
                if np.linalg.norm(m - mean) < 1e-12:
                    break

        return best_x, best_f
