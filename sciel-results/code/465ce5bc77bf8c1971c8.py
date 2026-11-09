import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        n_evals = 0

        def evaluate(x):
            nonlocal n_evals, best_x, best_f
            if n_evals >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Adaptive Simulated Annealing (ASA) with multi-start
        def asa_search(func_eval, x0, max_iter, T0=1.0, T_min=0.001, cooling=0.995):
            x = x0.copy()
            f = func_eval(x)
            T = T0
            rng = np.random.RandomState(int(hash(x.tobytes()) % (2**31 - 1)))

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if n_evals >= budget - 2:
                    break

                # Generate perturbation with adaptive step
                # Step size is proportional to current temperature
                step_size = T
                dx = rng.randn(dim) * step_size
                x_new = x + dx
                x_new = np.clip(x_new, -5, 5)

                f_new = func_eval(x_new)
                if n_evals >= budget:
                    break

                # Acceptance criterion
                if f_new < f:
                    x = x_new
                    f = f_new
                    # Slightly increase temperature on improvement to explore more
                    T = min(T0, T * 1.01)
                else:
                    # Metropolis acceptance
                    delta_f = f_new - f
                    if delta_f < 0:
                        accept = True
                    else:
                        accept = rng.rand() < np.exp(-delta_f / (T + 1e-10))

                    if accept:
                        x = x_new
                        f = f_new

                # Cool down
                T = max(T_min, T * cooling)

                # Early termination if temperature is very low and no improvement
                if T <= T_min * 1.01:
                    # Do a few more iterations with very small steps
                    for _ in range(min(5, max_iter - it)):
                        if n_evals >= budget - 1:
                            break
                        dx = rng.randn(dim) * 0.01
                        x_new = np.clip(x + dx, -5, 5)
                        f_new = func_eval(x_new)
                        if n_evals >= budget:
                            break
                        if f_new < f:
                            x = x_new
                            f = f_new
                    break

            return f

        # Nelder-Mead for polishing
        def nelder_mead(func_eval, x0, max_iter, initial_step=0.1):
            simplex = [x0.copy()]
            f_vals = [func_eval(x0)]

            for i in range(dim):
                if n_evals >= budget:
                    return min(f_vals) if f_vals else float('inf')
                x_new = x0.copy()
                x_new[i] += initial_step
                simplex.append(x_new)
                f_vals.append(func_eval(x_new))

            if len(f_vals) < dim + 1:
                return min(f_vals) if f_vals else float('inf')

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if n_evals >= budget - 5:
                    break

                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)

                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = func_eval(x_ref)
                if n_evals >= budget:
                    break

                if f_ref < f_vals[0]:
                    x_exp = x_centroid + gamma * (x_ref - x_worst)
                    f_exp = func_eval(x_exp)
                    if n_evals >= budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    f_con = func_eval(x_con)
                    if n_evals >= budget:
                        break
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = func_eval(simplex[i])
                            if n_evals >= budget:
                                break
                        if n_evals >= budget:
                            break

            return min(f_vals) if f_vals else float('inf')

        # Multi-start ASA with final NM polish
        n_restarts = 8
        rng_global = np.random.RandomState(42)

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break

            if r == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng_global.uniform(-4, 4, dim)

            # Determine iterations for this restart
            remaining = budget - n_evals
            if r < n_restarts - 1:
                # Allocate budget proportionally, leaving room for NM polish
                asa_iter = max(10, (remaining // (n_restarts - r)) * 2)
            else:
                asa_iter = max(10, remaining // 3)

            # ASA search
            asa_search(evaluate, x_start, asa_iter, T0=1.0, T_min=0.005, cooling=0.998)

            if n_evals >= budget:
                break

            # NM polish on best found so far
            if best_x is not None:
                remaining = budget - n_evals
                if remaining > 15:
                    nm_iter = max(5, remaining // 3)
                    nelder_mead(evaluate, best_x, nm_iter, initial_step=0.05)

        # Final polish with very tight NM
        if best_x is not None and n_evals < budget - 10:
            remaining = budget - n_evals
            if remaining > 10:
                nelder_mead(evaluate, best_x, max(5, remaining // 2), initial_step=0.01)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
