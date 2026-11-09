import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy:
        # 1. Coarse grid search to locate promising basins (Rastrigin has period ~2*pi, but local minima at integers).
        #    Since the domain is [-5, 5], and Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)),
        #    local minima are at integers. The global min is at the shifted origin (integer coords).
        #    A grid with step 1.0 covers all integer points. We evaluate a subset to find the best basin.
        # 2. CMA-ES (or (1+1)-ES with restarts) to refine within the basin.
        # 3. Nelder-Mead for final polish.

        # Phase 1: Coarse Grid Search
        # Points: integers from -4 to 4 (since min is in [-4,4]^5 per problem statement, but we check [-5,5] to be safe?
        # Problem says minimizer lies in [-4, 4]^5. So integers -4, -3, -2, -1, 0, 1, 2, 3, 4.
        # That's 9 points per dimension. 9^5 = 59049 points. Too many for 500 budget.
        # We need to be smarter.
        # Instead of full grid, we can use a random search or a coarse grid with fewer points.
        # Let's try a grid with step 2.0: -4, -2, 0, 2, 4. That's 5 points per dim. 5^5 = 3125. Still too many.
        # Step 2.5? -5, -2.5, 0, 2.5, 5.
        # Let's use a random search with 100 points, then pick the best 5 for local refinement.
        # Or, better: use a (1+1)-ES with large sigma to jump between basins.

        # Let's stick to the (1+1)-ES idea but make it robust.
        # The previous attempt had a messy loop. Let's write a clean (1+1)-ES with 1/5 rule.
        
        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(n)
        fx = evaluate(x)

        # If fx is very large, we might be far from min.
        # Let's do a few random restarts if we don't improve quickly.
        # But (1+1)-ES should handle it if sigma is large enough.
        
        sigma = 2.0  # Large initial step to cover the domain
        success_count = 0
        window = 10
        steps = 0

        # Budget allocation: 70% for ES, 30% for NM
        es_budget = int(budget * 0.7)

        while calls < es_budget:
            if time.time() > deadline - 5:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            fy = evaluate(y)
            
            steps += 1
            
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
            # else: keep x

            # Adapt sigma every 'window' steps
            if steps % window == 0:
                if success_count / window > 0.5:
                    sigma *= 1.5
                else:
                    sigma *= 0.5
                sigma = np.clip(sigma, 0.01, 5.0)
                success_count = 0

        # Phase 2: Nelder-Mead
        if best_x is None:
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        if calls < budget - 10 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            f_nm = best_f

            # Build initial simplex
            simplex = [x_nm.copy()]
            fvals = [f_nm]

            step_nm = 0.5
            for i in range(n):
                pert = x_nm.copy()
                if x_nm[i] + step_nm > 5.0:
                    pert[i] -= step_nm
                elif x_nm[i] - step_nm < -5.0:
                    pert[i] += step_nm
                else:
                    pert[i] += step_nm
                pert = np.clip(pert, -5.0, 5.0)
                
                f_pert = evaluate(pert)
                simplex.append(pert)
                fvals.append(f_pert)
                
                if calls >= budget:
                    break
            
            if len(simplex) < n + 1:
                return best_x, best_f

            # NM Iterations
            max_nm_iters = 100
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if np.std(fvals) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                fr = evaluate(xr)
                
                if calls >= budget:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = evaluate(xe)
                    if calls < budget:
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    beta = 0.5
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + beta * (xr - centroid), -5.0, 5.0)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - beta * (xr - centroid), -5.0, 5.0)
                    
                    fc = evaluate(xc)
                    
                    if calls >= budget:
                        break
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                            if calls >= budget:
                                break
            
            # Update best from NM
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        return best_x, best_f
