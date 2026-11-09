import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_iters=100, tol=1e-12):
            """
            Derivative-free Nelder-Mead optimization.
            Uses 6 calls per iteration for dim=5.
            """
            nonlocal n_calls, best_x, best_f
            
            # Initialize simplex
            # x0 is the initial best point
            # Create other vertices by perturbing x0
            simplex = []
            simplex.append(x0.copy())
            simplex_f = [call_f(x0)]
            
            if simplex_f[0] is None:
                return best_x, best_f

            # Standard initial simplex: perturb each dimension
            for i in range(dim):
                x_new = x0.copy()
                # Use a reasonable initial step size
                # Since box is [-5, 5], and minimizer is in [-4, 4], 
                # a step of 1.0 is a good start for exploration.
                if x_new[i] + 1.0 <= 5.0:
                    x_new[i] += 1.0
                else:
                    x_new[i] -= 1.0
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = call_f(x_new)
                if f_new is None:
                    return best_x, best_f
                simplex.append(x_new)
                simplex_f.append(f_new)

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for k in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if n_calls >= budget:
                    break

                # Sort simplex by function value
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                # Check convergence
                if simplex_f[-1] - simplex_f[0] < tol:
                    break

                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    return best_x, best_f

                if f_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        return best_x, best_f
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < simplex_f[-1] (which is true here)
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        return best_x, best_f
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = call_f(simplex[i])
                            if f_shrink is None:
                                return best_x, best_f
                            simplex_f[i] = f_shrink

            return best_x, best_f

        # Strategy:
        # 1. Start at 0 (center of box).
        # 2. Run Nelder-Mead.
        # 3. If budget remains, perform random restarts with smaller initial simplex or different starts.
        
        # Initial run
        nelder_mead(np.zeros(dim), max_iters=50)

        # Restart logic
        # NM is fast (6 calls/iter). With 500 budget, we can do ~80 iterations.
        # We used ~50. We have ~200 calls left.
        # We can do 2-3 more restarts.
        
        restarts = 0
        max_restarts = 3
        
        while restarts < max_restarts and n_calls < budget - 20 and time.time() < deadline - 10:
            # Generate a random start point
            # To avoid getting stuck in the same local minima, sample from a wider distribution
            # but keep it within the likely region [-4, 4]
            x_rand = rng.uniform(-4, 4, dim)
            
            # Run NM with a limited budget for this restart
            # Let's give it ~50% of remaining budget or a fixed reasonable amount
            remaining = budget - n_calls
            max_iters_this = min(30, remaining // 6)
            
            if max_iters_this < 5:
                break
                
            nelder_mead(x_rand, max_iters=max_iters_this)
            restarts += 1

        # Final polish: If we have a few calls left, do a very fine-grained local search
        # using a small step size around the best point found
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01 # Very small step
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    # sigma *= 1.1 # Grow slightly if improving
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-5)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
