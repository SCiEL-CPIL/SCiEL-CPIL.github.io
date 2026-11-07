"""Multi-start CMA-ES with adaptive budget allocation for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, max_iter, time_deadline, call_counter):
            """
            Run CMA-ES from x0 with initial sigma.
            Returns (best_x, best_f, calls_used)
            """
            dim = len(x0)
            x = x0.copy()
            f_x = func(x)
            call_counter[0] += 1
            record(x, f_x)

            sigma = sigma0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)  # We'll use a simplified approach
            C_inv_sqrt = np.eye(dim)

            # CMA-ES parameters
            mu = max(1, dim // 2)  # number of parents
            lambda_pop = 2 * dim  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1)) / ((dim + 1) ** 2 + mu_eff)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            best_x = x.copy()
            best_f = f_x

            for it in range(max_iter):
                if not time_ok() or not budget_ok():
                    break

                # Generate population
                ys = []
                xs = []
                fs = []
                for i in range(lambda_pop):
                    if not time_ok() or not budget_ok():
                        break
                    y = rng.standard_normal(dim)
                    # Use simplified covariance (identity for now, could be improved)
                    z = C_sqrt @ y
                    x_new = x + sigma * z
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    call_counter[0] += 1
                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_new)
                    record(x_new, f_new)

                if len(xs) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = [xs[i] for i in order]
                fs_sorted = [fs[i] for i in order]
                ys_sorted = [ys[i] for i in order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = xs_sorted[0].copy()

                # Update mean
                x_new_mean = np.sum([weights[i] * xs_sorted[i] for i in range(mu)], axis=0)

                # Update sigma
                y_mean = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / np.linalg.norm(C_sqrt)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 100.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x) / sigma
                C = (1 + c_1 + c_mu - c_c * (1 - c_c) - c_1 * mu_eff) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum([weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu)], axis=0)

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)

                x = x_new_mean

                # Check convergence
                if np.linalg.norm(x_new_mean - x) < 1e-12:
                    break

                # Update x
                x = x_new_mean

            return best_x, best_f, call_counter[0]

        # Multi-start strategy
        # Start 1: origin
        call_counter = [calls]
        bx1, bf1, c1 = cma_es(func, np.zeros(dim), 2.0, 150, deadline, call_counter)
        calls = call_counter[0]

        # Start 2: random point
        if time_ok() and budget_ok():
            x_start2 = rng.uniform(-3, 3, dim)
            call_counter = [calls]
            bx2, bf2, c2 = cma_es(func, x_start2, 1.5, 120, deadline, call_counter)
            calls = call_counter[0]
            if bf2 < best_f:
                best_f = bf2
                best_x = bx2.copy()

        # Start 3: another random point
        if time_ok() and budget_ok():
            x_start3 = rng.uniform(-4, 4, dim)
            call_counter = [calls]
            bx3, bf3, c3 = cma_es(func, x_start3, 1.0, 100, deadline, call_counter)
            calls = call_counter[0]
            if bf3 < best_f:
                best_f = bf3
                best_x = bx3.copy()

        # Final local refinement with Nelder-Mead if budget remains
        if best_x is not None and time_ok() and budget_ok():
            x0 = best_x.copy()
            f0 = best_f
            alpha = 0.2
            simplex = [x0.copy()]
            f_simplex = [f0]

            for i in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)

            if len(simplex) >= 2:
                gamma = 1.0
                rho = 2.0
                sigma_nm = 0.5
                shrink = 0.5
                phase_best_x = best_x.copy()
                phase_best_f = best_f

                for _ in range(200):  # Limit iterations
                    if not time_ok() or not budget_ok():
                        break

                    indices = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in indices]
                    f_simplex = [f_simplex[i] for i in indices]

                    x_best = simplex[0]
                    f_best = f_simplex[0]
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    x_second_worst = simplex[-2]
                    f_second_worst = f_simplex[-2]

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_r = func(x_r)
                    calls += 1
                    record(x_r, f_r)
                    if f_r < phase_best_f:
                        phase_best_f = f_r
                        phase_best_x = x_r.copy()

                    if f_r < f_best:
                        x_e = centroid + rho * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if not time_ok() or not budget_ok():
                            break
                        f_e = func(x_e)
                        calls += 1
                        record(x_e, f_e)
                        if f_e < phase_best_f:
                            phase_best_f = f_e
                            phase_best_x = x_e.copy()
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_second_worst:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        if f_r < f_worst:
                            x_c = centroid + sigma_nm * (x_r - centroid)
                        else:
                            x_c = centroid - sigma_nm * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)

                        if not time_ok() or not budget_ok():
                            break
                        f_c = func(x_c)
                        calls += 1
                        record(x_c, f_c)
                        if f_c < phase_best_f:
                            phase_best_f = f_c
                            phase_best_x = x_c.copy()

                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            for i in range(1, len(simplex)):
                                if not time_ok() or not budget_ok():
                                    break
                                simplex[i] = x_best + shrink * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                                record(simplex[i], f_simplex[i])
                                if f_simplex[i] < phase_best_f:
                                    phase_best_f = f_simplex[i]
                                    phase_best_x = simplex[i].copy()
                            if not time_ok():
                                break

                    f_spread = np.max(f_simplex) - np.min(f_simplex)
                    if f_spread < 1e-12:
                        break

                if phase_best_f < best_f:
                    best_f = phase_best_f
                    best_x = phase_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
