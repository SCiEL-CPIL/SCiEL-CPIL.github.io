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
        
        # Ensure we have n+1 points
        if len(simplex) < n + 1:
             # Fill with random if evaluation failed or budget hit
             while len(simplex) < n + 1 and calls < budget:
                 x_rand = x_start + np.random.uniform(-1, 1, n)
                 x_rand = np.clip(x_rand, -5, 5)
                 f_rand = evaluate(x_rand)
                 simplex.append((f_rand, x_rand))
        
        # Nelder-Mead Main Loop
        while calls < budget and time_left() > 0:
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

        # Phase 3: Newton's Method with Finite Difference Hessian
        # Use the best point found so far as the starting point
        x = best_x.copy() if best_x is not None else np.zeros(n)
        f = best_f
        
        # If we haven't made any calls yet, make sure we have a valid starting point
        if calls == 0:
            f = evaluate(x)
        
        h = 1e-3
        c1 = 1e-4
        rho_newton = 0.5

        while calls < budget and time_left() > 0:
            # Estimate Gradient and Hessian
            g = np.zeros(n)
            H = np.zeros((n, n))
            
            f0 = f
            for i in range(n):
                if calls >= budget or time_left() <= 0: 
                    break
                
                # Central difference for Gradient
                x_plus = x.copy()
                x_plus[i] += h
                fx_plus = func(x_plus)
                calls += 1
                if fx_plus < best_f:
                    best_f = fx_plus
                    best_x = x_plus.copy()
                
                if calls >= budget or time_left() <= 0: break
                x_minus = x.copy()
                x_minus[i] -= h
                fx_minus = func(x_minus)
                calls += 1
                if fx_minus < best_f:
                    best_f = fx_minus
                    best_x = x_minus.copy()
                
                g[i] = (fx_plus - fx_minus) / (2 * h)
                
                # Second derivative for Hessian diagonal
                H[i, i] = (fx_plus - 2 * f0 + fx_minus) / (h ** 2)
                
                # Cross derivatives for Hessian off-diagonal
                for j in range(i + 1, n):
                    if calls >= budget or time_left() <= 0: break
                    x_ij_plus_plus = x.copy()
                    x_ij_plus_plus[i] += h
                    x_ij_plus_plus[j] += h
                    fx_ij_pp = func(x_ij_plus_plus)
                    calls += 1
                    if fx_ij_pp < best_f:
                        best_f = fx_ij_pp
                        best_x = x_ij_plus_plus.copy()
                    
                    if calls >= budget or time_left() <= 0: break
                    x_ij_plus_minus = x.copy()
                    x_ij_plus_minus[i] += h
                    x_ij_plus_minus[j] -= h
                    fx_ij_pm = func(x_ij_plus_minus)
                    calls += 1
                    if fx_ij_pm < best_f:
                        best_f = fx_ij_pm
                        best_x = x_ij_plus_minus.copy()
                    
                    if calls >= budget or time_left() <= 0: break
                    x_ij_minus_plus = x.copy()
                    x_ij_minus_plus[i] -= h
                    x_ij_minus_plus[j] += h
                    fx_ij_mp = func(x_ij_minus_plus)
                    calls += 1
                    if fx_ij_mp < best_f:
                        best_f = fx_ij_mp
                        best_x = x_ij_minus_plus.copy()
                    
                    if calls >= budget or time_left() <= 0: break
                    x_ij_minus_minus = x.copy()
                    x_ij_minus_minus[i] -= h
                    x_ij_minus_minus[j] -= h
                    fx_ij_mm = func(x_ij_minus_minus)
                    calls += 1
                    if fx_ij_mm < best_f:
                        best_f = fx_ij_mm
                        best_x = x_ij_minus_minus.copy()
                    
                    H[i, j] = (fx_ij_pp - fx_ij_pm - fx_ij_mp + fx_ij_mm) / (4 * h ** 2)
                    H[j, i] = H[i, j]

            if calls >= budget or time_left() <= 0:
                break

            # Regularize Hessian if not positive definite
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) < 1e-6:
                H = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T

            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g

            # Line Search
            alpha_ls = 1.0
            slope = g @ p
            if slope >= 0:
                p = -p
                slope = -slope
                if slope <= 0:
                    break
            
            f_new = f
            x_new = x.copy()
            
            for _ in range(20):
                if calls >= budget or time_left() <= 0: break
                x_trial = x + alpha_ls * p
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                
                if f_trial <= f + c1 * alpha_ls * slope:
                    f_new = f_trial
                    x_new = x_trial
                    break
                else:
                    alpha_ls *= rho_newton
            
            if calls >= budget or time_left() <= 0:
                break

            x = x_new
            f = f_new

            if alpha_ls < 1e-6 or np.linalg.norm(g) < 1e-8:
                break

        return best_x, best_f
