import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        deadline = time.time() + 290.0
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES with adaptive step size
        x = np.zeros(n)
        fx = eval_point(x)
        sigma = 2.0
        success_count = 0
        total_steps = 0

        es_budget = int(self.budget * 0.6)
        while calls_made < es_budget and time.time() < deadline:
            z = rng.standard_normal(n)
            x_trial = x + sigma * z
            fx_trial = eval_point(x_trial)

            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                success_count += 1
            else:
                success_count -= 1

            total_steps += 1
            if total_steps % 10 == 0:
                # Adapt sigma based on success rate over last 10 steps
                # Simple adaptation: increase if many successes, decrease if many failures
                if success_count > 2:
                    sigma *= 1.2
                elif success_count < -2:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.01, 10.0)
                success_count = 0

        # Phase 2: Nelder-Mead local refinement
        nm_budget = self.budget - calls_made
        if nm_budget > n + 2 and time.time() < deadline:
            x0 = best_x.copy()
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
                eval_point(xi)

            simplex = np.array(simplex)
            fvals = np.array([eval_point(simplex[i]) for i in range(n + 1)])
            # Note: the above already evaluated, so we need to track properly
            # Let's redo Nelder-Mead more carefully

        # Actually, let's do a cleaner Nelder-Mead
        calls_before_nm = calls_made
        remaining = self.budget - calls_before_nm
        if remaining > n + 10 and time.time() < deadline:
            # Initialize simplex around best point
            x0 = best_x.copy()
            simplex = [x0.copy()]
            step = 0.1
            for i in range(n):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi)

            simplex = np.array(simplex)
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                fvals[i] = eval_point(simplex[i])

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            while calls_made < self.budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                if calls_made >= self.budget or time.time() > deadline:
                    break
                f_r = eval_point(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    f_e = eval_point(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    f_c = eval_point(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            fvals[i] = eval_point(simplex[i])

        return best_x, best_f
