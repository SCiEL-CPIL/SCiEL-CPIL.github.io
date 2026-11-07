"""Combined Nelder-Mead with adaptive restarts and local polish for BBOB f13."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Initialize simplex at the center (0,0,0,0,0)
        # This follows the directive to use NM from center
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        if calls >= budget:
            return best_x, best_fx

        # Create initial simplex
        # Standard NM simplex: x0, x0 + delta*ei
        delta = 0.5
        simplex = [x0.copy()]
        fx_list = [f0]
        
        for i in range(dim):
            if calls >= budget:
                break
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            fx_list.append(evaluate(xi))
            
        n_simplex = len(simplex)
        if n_simplex < 2:
            return best_x, best_fx

        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        stall_count = 0
        max_stall = 20
        prev_best = best_fx

        # Main NM Loop
        while calls < budget and time_ok():
            # Sort simplex
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_fx = fx_list[-1]
            best_fx_simplex = fx_list[0]

            # 1. Reflection
            xr = centroid + alpha * (centroid - worst)
            
            if calls >= budget: break
            fr = evaluate(xr)
            
            if fr < best_fx_simplex:
                # 2. Expansion
                if calls < budget:
                    xe = centroid + gamma * (centroid - worst)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fx_list[-1] = fe
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fr
            elif fr < worst_fx:
                simplex[-1] = xr
                fx_list[-1] = fr
            else:
                # 3. Contraction
                if fr < worst_fx:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (worst - centroid)
                
                if calls >= budget: break
                fc = evaluate(xc)
                
                if fc < worst_fx:
                    simplex[-1] = xc
                    fx_list[-1] = fc
                else:
                    # 4. Shrink
                    for i in range(1, n_simplex):
                        if calls >= budget or not time_ok():
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fx_list[i] = evaluate(simplex[i])
            
            # Check convergence
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.all(spread < 1e-8):
                break

            # Stagnation check
            if best_fx < prev_best - 1e-6:
                stall_count = 0
                prev_best = best_fx
            else:
                stall_count += 1

            if stall_count > max_stall:
                # Try a random restart if we have budget left
                if calls + 10 < budget and time_ok():
                    x_new = rng.uniform(-4, 4, dim)
                    fx_new = evaluate(x_new)
                    if fx_new < best_fx:
                        # Re-initialize simplex around new best
                        x0 = best_x.copy()
                        f0 = best_fx
                        simplex = [x0.copy()]
                        fx_list = [f0]
                        for i in range(dim):
                            if calls >= budget: break
                            xi = x0.copy()
                            xi[i] += 0.5
                            simplex.append(xi)
                            fx_list.append(evaluate(xi))
                        n_simplex = len(simplex)
                        stall_count = 0
                        prev_best = best_fx
                    else:
                        break
                else:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
