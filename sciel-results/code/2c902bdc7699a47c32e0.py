import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with restarts for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

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

        def nelder_mead(start_x, max_evals):
            nonlocal eval_count
            if eval_count >= budget:
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            tol = 1e-8

            x0 = np.clip(start_x, -5, 5)
            f0 = eval_func(x0)
            if eval_count >= budget:
                return

            simplex = [x0]
            fs = [f0]
            
            step = 0.05
            for i in range(d):
                xi = x0.copy()
                xi[i] += step
                fi = eval_func(xi)
                simplex.append(xi)
                fs.append(fi)
                if eval_count >= budget:
                    return

            def sort_simplex():
                nonlocal simplex, fs
                indices = np.argsort(fs)
                simplex = [simplex[i] for i in indices]
                fs = [fs[i] for i in indices]

            sort_simplex()
            last_best = fs[0]
            stagnation_count = 0

            while eval_count < budget and (budget - eval_count) > max_evals * 0.1: # Stop if little budget left
                if fs[-1] - fs[0] < tol:
                    break

                if fs[0] < last_best:
                    last_best = fs[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 20: # 20 iterations without improvement
                        break

                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fs[-1]

                x_r = centroid + alpha * (centroid - x_worst)
                f_r = eval_func(x_r)
                if eval_count >= budget:
                    break

                if f_r < fs[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if eval_count >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    x_c = centroid + rho * (x_worst - centroid)
                    f_c = eval_func(x_c)
                    if eval_count >= budget:
                        break
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = eval_func(simplex[i])
                            if eval_count >= budget:
                                break
                        if eval_count >= budget:
                            break

                sort_simplex()

        # Start from origin
        nelder_mead(np.zeros(d), budget)

        # If budget remains, try a few random restarts
        if eval_count < budget - 10:
            for _ in range(5):
                if eval_count >= budget - 10:
                    break
                x_rand = np.random.uniform(-5, 5, d)
                nelder_mead(x_rand, 50) # Limit each restart to ~50 evals

        return best_x, best_f
