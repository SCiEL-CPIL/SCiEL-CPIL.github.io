"""Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and time.time() < deadline

        # Phase 1: Halton quasi-random initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            points = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n + 1):
                p = np.array([halton(i, bases[d]) for d in range(dim)])
                p = p * 10.0 - 5.0  # map [0,1] to [-5,5]
                points.append(p)
            return np.array(points)

        n_init = 20
        init_points = halton_sequence(n_init, dim)

        init_results = []
        for i in range(len(init_points)):
            if not time_ok():
                break
            x = init_points[i]
            f = evaluate(x)
            init_results.append((x.copy(), f))

        init_results.sort(key=lambda x: x[1])

        # Phase 2: Multi-start (1+1)-ES on top candidates
        K = min(4, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(30, remaining_budget // K)

        for x0, f0 in top_starts:
            if not time_ok():
                break

            x = x0.copy()
            fx = f0
            sigma = 0.5

            # 1/5 success rule parameters
            n_success = 0
            n_trials = 0
            max_trials = budget_per_start

            for trial in range(max_trials):
                if not time_ok():
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                n_trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    n_success += 1
                    sigma = min(sigma * 1.2, 3.0)
                else:
                    sigma = max(sigma * 0.8, 0.01)

                # 1/5 success rule adaptation
                if n_trials % 5 == 0:
                    if n_success / 5.0 > 1.0 / 5.0:
                        sigma *= 1.1
                    elif n_success / 5.0 < 1.0 / 5.0:
                        sigma *= 0.9
                    sigma = max(0.01, min(sigma, 3.0))
                    n_success = 0

                # Check convergence
                if sigma < 0.02:
                    break

            # Nelder-Mead polish on the best found so far
            if not time_ok():
                break

            nm_x = best_x.copy() if best_x is not None else x0.copy()
            nm_f = best_f
            nm_sigma = 0.1

            # Nelder-Mead simplex
            simplex = [nm_x.copy()]
            for d in range(dim):
                p = nm_x.copy()
                p[d] += nm_sigma
                simplex.append(p)

            simplex = np.array(simplex)
            fvals = [nm_f]
            for i in range(1, dim + 1):
                if not time_ok():
                    break
                fvals.append(evaluate(simplex[i]))

            # Nelder-Mead iterations
            max_nm_iter = 25
            for it in range(max_nm_iter):
                if not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = evaluate(xr)

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])

                # Check convergence
                if max(fvals) - min(fvals) < 1e-8:
                    break

        # Phase 3: Final fine-tuning around global best
        if time_ok() and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.02

            for _ in range(40):
                if not time_ok():
                    break
                y = x_polish + sigma_p * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p = max(0.005, sigma_p * 0.95)
                else:
                    sigma_p = max(0.002, sigma_p * 0.9)

        return best_x, best_f
# EVOLVE-BLOCK-END
