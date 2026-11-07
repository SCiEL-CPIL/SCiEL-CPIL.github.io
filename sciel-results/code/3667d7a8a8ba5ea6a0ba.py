"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with coordinate-wise adaptive step sizes and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        # Global best tracking
        best_x = np.zeros(dim)
        best_fx = np.inf

        # Weierstrass f16 parameters (BBOB standard)
        # f(x) = sum_{k=0}^{K} a_k * sum_{j=0}^{inf} b_k^j * cos(2*pi*b_k^j*(x_j + c_k)) - K*a_0
        # BBOB f16: a_k = (1/2)^k, b_k = 0.5 to 0.9, c_k in [0,1]
        # For optimization, we just use the black-box function.

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Initial evaluation at center
        x0 = np.zeros(dim)
        fx0 = func(x0)
        update_best(x0, fx0)
        evals_used = 1

        # Multi-start: generate diverse starting points
        rng = np.random.default_rng(12345)
        starts = [np.zeros(dim)]
        # Add random points spread across the domain
        for _ in range(5):
            pt = rng.uniform(-4, 4, size=dim)
            starts.append(pt)

        # CMA-ES with coordinate-wise adaptive step sizes (CMA-ES-lite)
        # Each dimension has its own step size, updated based on success rate

        def cmaes_lite(start_point, eval_budget, time_budget):
            nonlocal best_x, best_fx, evals_used

            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))  # lam = 8 for n=5
            mu = lam // 2  # mu = 4

            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / (weights ** 2).sum()

            # Initial mean and per-dimension step sizes
            mean = start_point.copy()
            sigmas = np.full(n, 2.0)  # per-dimension step sizes

            # Evolution path for step size adaptation (per dimension)
            ps = np.zeros(n)
            cs = (n + 1.3) / (n + 3 + 2 * n / (n - 1))
            damps = 1 + 2 * max(0, np.sqrt((n + 1) / (n - 1) - 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track success per dimension for adaptive step sizes
            success_counts = np.zeros(n)
            total_counts = np.zeros(n)
            window_size = 20

            local_best = start_point.copy()
            local_best_fx = func(start_point)
            evals_used += 1
            if local_best_fx < best_fx:
                best_fx = local_best_fx
                best_x = local_best.copy()

            # Main loop
            while evals_used < eval_budget and (time.time() - start_time) < time_budget:
                # Generate population
                z = rng.standard_normal((lam, n))
                # Apply per-dimension step sizes
                y = mean[None, :] + z * sigmas[None, :]
                y = np.clip(y, -5, 5)

                # Evaluate
                fvals = np.empty(lam)
                for i in range(lam):
                    if evals_used >= eval_budget:
                        break
                    fvals[i] = func(y[i])
                    evals_used += 1

                # Sort
                order = np.argsort(fvals)
                y_sorted = y[order]
                fvals_sorted = fvals[order]

                # Update local best
                if fvals_sorted[0] < local_best_fx:
                    local_best_fx = fvals_sorted[0]
                    local_best = y_sorted[0].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best.copy()

                # Update mean using weighted recombination
                new_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

                # Update evolution path
                step = (new_mean - mean) / (np.prod(sigmas) ** 0.5)  # normalized step
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * step

                # Update per-dimension step sizes based on success rate
                # A move is "successful" if it improves the best
                for i in range(mu):
                    for d in range(n):
                        if y_sorted[i, d] != mean[d]:
                            total_counts[d] += 1
                            if fvals_sorted[i] < fvals_sorted[i + 1] if i + 1 < lam else True:
                                success_counts[d] += 1

                # Adapt step sizes based on success rate in recent window
                for d in range(n):
                    if total_counts[d] > window_size:
                        success_rate = success_counts[d] / total_counts[d]
                        # Increase step size if success rate is high, decrease if low
                        if success_rate > 0.6:
                            sigmas[d] *= 1.1
                        elif success_rate < 0.3:
                            sigmas[d] *= 0.9
                        # Reset counters
                        success_counts[d] = 0
                        total_counts[d] = 0

                # Also adapt based on evolution path
                norm_ps = np.linalg.norm(ps)
                for d in range(n):
                    sigmas[d] *= np.exp(cs / damps * (norm_ps / chi_n - 1))
                    sigmas[d] = np.clip(sigmas[d], 1e-6, 50.0)

                mean = new_mean

            return local_best, local_best_fx

        # Run CMA-ES-lite from multiple starts
        # First start (center) gets more budget
        evals_for_first = min(200, budget - 1)
        time_for_first = 150.0

        if evals_for_first > 1:
            lb, lbfx = cmaes_lite(starts[0], evals_used + evals_for_first, time_for_first)
            update_best(lb, lbfx)

        # Remaining starts get less budget each
        remaining_starts = starts[1:]
        if remaining_starts and evals_used < budget - 1:
            budget_per_start = max(30, (budget - evals_used - 1) // len(remaining_starts))
            time_per_start = 60.0

            for start_pt in remaining_starts:
                if evals_used >= budget - 1 or (time.time() - start_time) > time_limit - 30:
                    break
                lb, lbfx = cmaes_lite(start_pt, evals_used + budget_per_start, time_per_start)
                update_best(lb, lbfx)

        # Local polish using coordinate descent with adaptive step sizes
        if evals_used < budget - 5 and (time.time() - start_time) < time_limit - 10:
            x = best_x.copy()
            fx = best_fx
            step_size = 0.1

            # Coordinate descent with line search
            for iteration in range(100):
                if evals_used >= budget - 2 or (time.time() - start_time) > time_limit - 5:
                    break

                improved = False
                for d in range(dim):
                    if evals_used >= budget - 2 or (time.time() - start_time) > time_limit - 5:
                        break

                    # Try moving in positive direction
                    x_plus = x.copy()
                    x_plus[d] = min(5.0, x[d] + step_size)
                    if x_plus[d] != x[d]:
                        fx_plus = func(x_plus)
                        evals_used += 1
                        if fx_plus < fx:
                            x = x_plus
                            fx = fx_plus
                            improved = True
                            update_best(x, fx)
                            # Try to expand step
                            for _ in range(3):
                                if evals_used >= budget - 2:
                                    break
                                x_exp = x.copy()
                                x_exp[d] = min(5.0, x[d] + step_size)
                                if x_exp[d] != x[d]:
                                    fx_exp = func(x_exp)
                                    evals_used += 1
                                    if fx_exp < fx:
                                        x = x_exp
                                        fx = fx_exp
                                        step_size *= 1.5
                                        update_best(x, fx)
                                    else:
                                        break
                                else:
                                    break
                        else:
                            # Try negative direction
                            x_minus = x.copy()
                            x_minus[d] = max(-5.0, x[d] - step_size)
                            if x_minus[d] != x[d]:
                                fx_minus = func(x_minus)
                                evals_used += 1
                                if fx_minus < fx:
                                    x = x_minus
                                    fx = fx_minus
                                    improved = True
                                    update_best(x, fx)
                                    # Try to expand step in negative direction
                                    for _ in range(3):
                                        if evals_used >= budget - 2:
                                            break
                                        x_exp = x.copy()
                                        x_exp[d] = max(-5.0, x[d] - step_size)
                                        if x_exp[d] != x[d]:
                                            fx_exp = func(x_exp)
                                            evals_used += 1
                                            if fx_exp < fx:
                                                x = x_exp
                                                fx = fx_exp
                                                step_size *= 1.5
                                                update_best(x, fx)
                                            else:
                                                break
                                        else:
                                            break

                if not improved:
                    # Shrink step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

        # Final Nelder-Mead polish if budget remains
        if evals_used < budget - 10 and (time.time() - start_time) < time_limit - 5:
            x0 = best_x.copy()
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                delta = 0.05 if abs(x0[i]) < 4 else -0.05
                xi[i] += delta
                xi[i] = np.clip(xi[i], -5, 5)
                simplex.append(xi)

            fvals_nm = np.array([func(pt) for pt in simplex])
            evals_used += dim + 1

            if evals_used < budget - 5:
                alpha, gamma, rho = 1.0, 2.0, 0.5
                max_iter = min(budget - evals_used - 2, 100)

                for _ in range(max_iter):
                    if evals_used >= budget - 1 or (time.time() - start_time) > time_limit - 2:
                        break

                    order = np.argsort(fvals_nm)
                    simplex = [simplex[i] for i in order]
                    fvals_nm = fvals_nm[order]

                    if fvals_nm[-1] - fvals_nm[0] < 1e-12:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = func(xr)
                    evals_used += 1

                    if fr < fvals_nm[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = func(xe)
                        evals_used += 1
                        if fe < fr:
                            simplex[-1], fvals_nm[-1] = xe, fe
                        else:
                            simplex[-1], fvals_nm[-1] = xr, fr
                    elif fr < fvals_nm[-2]:
                        simplex[-1], fvals_nm[-1] = xr, fr
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        evals_used += 1
                        if fc < fvals_nm[-1]:
                            simplex[-1], fvals_nm[-1] = xc, fc
                        else:
                            for i in range(1, dim + 1):
                                if evals_used >= budget - 1:
                                    break
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals_nm[i] = func(simplex[i])
                                evals_used += 1

                    min_idx = np.argmin(fvals_nm)
                    update_best(simplex[min_idx], fvals_nm[min_idx])

        return best_x, best_fx
# EVOLVE-BLOCK-END
