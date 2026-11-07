import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline or calls_used >= budget

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if check_deadline():
                return np.inf
            x = np.clip(x, -5, 5)
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_calls):
            nonlocal calls_used
            if check_deadline():
                return
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [evaluate(x0)]
            
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += 0.5
                simplex.append(x_i)
                f_values.append(evaluate(x_i))
            
            # If we ran out of calls during initialization, stop
            if check_deadline():
                return

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for _ in range(max_calls):
                if check_deadline():
                    break

                # Sort simplex by fitness
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)

                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = evaluate(simplex[i])
                        if check_deadline():
                            break

        def run_1plus1_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            if check_deadline():
                return
            x = x0.copy()
            fx = evaluate(x)
            if check_deadline():
                return
            
            sigma = sigma0
            no_improve = 0
            max_no_improve = 20 # Restart sooner to allow more restarts

            for _ in range(max_calls):
                if check_deadline():
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                fy = evaluate(y)
                if check_deadline():
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.9
                    if no_improve >= max_no_improve:
                        break

        # Phase 1: Nelder-Mead from diverse starting points
        # 25 starts, 20 calls each = 500 calls. 
        # We will use a bit less to leave room for polish if needed, 
        # but the directive says "fixed budget of 20 function evaluations to each".
        # 25 * 20 = 500. This uses the whole budget.
        # However, NM initialization costs n+1 calls. So 20 calls per start is tight.
        # Let's stick to the directive: 25 starts, 20 calls each.
        
        n_starts = 25
        calls_per_start = 20
        
        # Generate diverse starting points
        # Use a quasi-random sequence (Halton-like or just random) for diversity
        # We'll use random points for simplicity and robustness against rotation
        start_points = []
        for i in range(n_starts):
            # Mix of origin, axis-aligned, and random
            if i == 0:
                p = np.zeros(dim)
            elif i <= dim:
                p = np.zeros(dim)
                p[i-1] = 3.0
            else:
                p = rng.uniform(-4, 4, dim)
            start_points.append(p)
            
        for p in start_points:
            if check_deadline():
                break
            nelder_mead(p, calls_per_start)
        
        # If we have budget left (e.g., if NM finished early due to convergence or deadline check inside),
        # do a final polish with (1+1)-ES from the best point found.
        if not check_deadline() and best_x is not None:
            remaining = budget - calls_used
            if remaining > 10:
                run_1plus1_es(best_x, 0.5, remaining)

        return best_x, best_f
