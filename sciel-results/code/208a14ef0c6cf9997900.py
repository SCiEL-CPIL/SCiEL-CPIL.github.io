"""Hybrid optimizer: Multi-start exploration + CMA-ES + Nelder-Mead for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Multi-start exploration + CMA-ES + Nelder-Mead for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != dim:
                x = np.resize(x, dim)
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_iter=60, initial_step=0.3):
            """Nelder-Mead local refinement."""
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return x0, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            f0 = safe_func(x0)
            fvals = [f0]

            for i in range(dim):
                xi = x0.copy()
                xi[i] += initial_step
                simplex.append(xi)
                fvals.append(safe_func(simplex[i]))

            alpha, gamma, rho = 1.0, 2.0, 0.5

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    fc = safe_func(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])

                if calls >= budget or time.time() > deadline:
                    break

            return best_x, best_f

        def run_cma_es(x_start, sigma_start, max_calls):
            """Run CMA-ES starting from x_start with given sigma for at most max_calls additional calls."""
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() > deadline:
                return

            rng = np.random.default_rng(42)

            lambda_ = 10
            mu = 5

            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size adaptation (CSA)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Chi-n for Hoeffding test
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initialize CMA-ES
            x_mean = x_start.copy()
            sigma = sigma_start
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)

            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Generate population
                pop = np.zeros((lambda_, dim))
                fvals = np.zeros(lambda_)

                for i in range(lambda_):
                    if calls >= max_calls or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    pop[i] = np.clip(x_mean + sigma * y, -5, 5)
                    fvals[i] = safe_func(pop[i])

                if calls >= max_calls or time.time() > deadline:
                    break

                # Sort by fitness
                indices = np.argsort(fvals)
                pop_sorted = pop[indices]

                # Weighted recombination
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * pop_sorted[i]

                # Update evolution paths
                y_diff = (x_new - x_mean) / sigma
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff

                # Update step size
                cs_norm = np.linalg.norm(pc) / chi_n
                hsig = (cs_norm < (1.4 + 1 / dim) * chi_n)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance matrix
                if hsig:
                    delta_hs = (1 - c_c) / c_c
                else:
                    delta_hs = 1
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (pop_sorted[i] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c_c * (delta_hs - 1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

                # Eigen decomposition
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                    B = B2
                    D = D2
                except np.linalg.LinAlgError:
                    pass

                x_mean = x_new

                # Stagnation check
                if sigma < 0.005:
                    break

        # Phase 1: Multi-start exploration - sample diverse points to find good basins
        rng = np.random.default_rng(42)
        n_initial = 15
        candidates = []
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of random points and structured grid-like points
            if i < 5:
                # Corner/edge points
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < 10:
                # Center-biased points
                x = rng.uniform(-2.0, 2.0, dim)
            else:
                # Random
                x = rng.uniform(-5.0, 5.0, dim)
            f = safe_func(x)
            candidates.append((f, x.copy()))

        # Sort candidates by fitness and keep top 3
        candidates.sort(key=lambda t: t[0])
        top_candidates = [c[1] for c in candidates[:3]]

        # Phase 2: CMA-ES on the best candidate
        # Use more of the budget for CMA-ES, leave ~60 for Nelder-Mead
        max_cma_calls = budget - 60

        run_cma_es(top_candidates[0], 1.5, max_cma_calls)

        # Phase 3: If we have budget left, try CMA-ES on second best candidate briefly
        if calls < max_cma_calls - 20 and len(top_candidates) > 1 and time.time() < deadline:
            remaining_for_cma2 = min(30, max_cma_calls - calls)
            if remaining_for_cma2 > 15:
                run_cma_es(top_candidates[1], 1.0, calls + remaining_for_cma2)

        # Phase 4: Nelder-Mead on the best point found
        if calls < budget and time.time() < deadline:
            nelder_mead(best_x.copy(), max_iter=50, initial_step=0.2)

        # Phase 5: Final polish if budget remains
        if calls < budget and time.time() < deadline:
            # Very fine-grained Nelder-Mead for final convergence
            nelder_mead(best_x.copy(), max_iter=30, initial_step=0.05)

        return best_x, best_f
# EVOLVE-BLOCK-END
