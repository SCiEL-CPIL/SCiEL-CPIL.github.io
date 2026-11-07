import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            if time.time() > deadline or calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_calls):
            nonlocal calls_used
            n = dim
            # Initial simplex
            alpha = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += alpha
                simplex.append(x_i)
            
            # Evaluate initial simplex
            f_vals = []
            for x in simplex:
                f_vals.append(evaluate(x))
                if calls_used >= max_calls:
                    break
            
            if calls_used >= max_calls:
                return

            # Nelder-Mead parameters
            r = 1.0   # reflection
            e = 2.0   # expansion
            c = 0.5   # contraction
            s = 0.5   # shrink
            
            while calls_used < max_calls and time.time() < deadline:
                # Sort simplex by function value
                indices = np.argsort(f_vals)
                simplex = [simplex[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                f_worst = f_vals[-1]
                f_best = f_vals[0]
                
                # Check convergence
                if np.max(f_vals) - np.min(f_vals) < 1e-8:
                    break
                if np.linalg.norm(simplex[-1] - simplex[0]) < 1e-8:
                    break

                # Centroid of best n points
                centroid = np.sum(simplex[:-1], axis=0) / n
                
                # Reflection
                x_r = centroid + r * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if calls_used >= max_calls:
                    break
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + e * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if calls_used >= max_calls:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + c * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if calls_used >= max_calls:
                            break
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = x_best + s * (simplex[i] - x_best)
                                f_vals[i] = evaluate(simplex[i])
                                if calls_used >= max_calls:
                                    return
                    else:
                        # Inside contraction
                        x_c = centroid - c * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if calls_used >= max_calls:
                            break
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = x_best + s * (simplex[i] - x_best)
                                f_vals[i] = evaluate(simplex[i])
                                if calls_used >= max_calls:
                                    return

        # Strategy: Multi-Start Nelder-Mead
        # 1. Origin
        # 2. 4 Random points
        # Allocate budget: 5 starts * 80 calls = 400 calls.
        # Remaining 100 calls for final polish on best.
        
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
            
        calls_per_start = 80
        
        for x0 in starts:
            if calls_used >= budget:
                break
            nelder_mead(x0, calls_per_start)
            
        # Final polish
        if best_x is not None and calls_used < budget:
            remaining = budget - calls_used
            # Use a smaller alpha for the final run? Or just run Nelder-Mead again.
            # Nelder-Mead might have converged, but restarting with a fresh simplex at the best point
            # with a slightly smaller initial step can help if it got stuck.
            x_final = best_x.copy()
            # Shrink initial simplex size for refinement
            # We can't easily change alpha inside nelder_mead without refactoring.
            # Let's just run it again.
            nelder_mead(x_final, remaining)

        return best_x, best_f
