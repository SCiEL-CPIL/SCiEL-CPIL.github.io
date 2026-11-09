"""Improved BBOB f22 optimizer: LHS + Multi-Start (1+1)-ES + CMA-ES + NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for better initial coverage
        n_lhs = min(50, max(20, budget // 10))
        points = []
        for i in range(n_lhs):
            if not check_time() or calls_made >= budget:
                break
            # LHS: divide each dimension into n_lhs equal intervals, sample one point per interval
            x = np.zeros(dim)
            for d in range(dim):
                interval = i // (n_lhs // dim + 1)
                # Actually, simpler: just use stratified sampling
                x[d] = -5.0 + (i + rng.random()) * (10.0 / n_lhs)
            x = np.clip(x, -5, 5)
            fx = safe_func(x)
            points.append((fx, x.copy()))

        if not points:
            return best_x, best_f

        points.sort(key=lambda t: t[0])

        # (1+1)-ES with 1/5 success rule
        def one_plus_one_es(x0, f0, max_steps=60):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (budget - calls_made) <= 0:
                return

            sigma = 0.5
            successes = 0
            trials = 0
            x_curr = x0.copy()
            f_curr = f0

            for _ in range(max_steps):
                if not check_time() or (budget - calls_made) <= 0:
                    break

                y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                trials += 1

                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    successes += 1

                # 1/5 success rule
                if trials >= 5:
                    if successes / trials > 0.2:
                        sigma *= 1.25
                    else:
                        sigma *= 0.8
                    successes = 0
                    trials = 0

                sigma = np.clip(sigma, 1e-6, 10.0)
                if sigma < 1e-5:
                    break

        # CMA-ES implementation
        def cma_es(x0, f0, max_steps=40):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (budget - calls_made) <= 0:
                return

            # CMA-ES parameters
            pop_size = 4  # Small population for limited budget
            sigma = 0.3
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (pop_size + 2) / (dim + pop_size + 2)
            cmu = min(0.2, (pop_size - 2 + 2.0 / (pop_size + 2)) / ((dim + 2) * (dim + 2)))
            cc = (4 + dim / 4) / (dim + 4 + 2 * cs)
            cm1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            cm2 = min(1 - cm1, 2 / ((dim + 2) ** 2 + pop_size))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            x_mean = x0.copy()
            f_mean = f0
            best_local_x = x0.copy()
            best_local_f = f0

            for step in range(max_steps):
                if not check_time() or (budget - calls_made) <= 0:
                    break

                # Generate population
                offspring = []
                for i in range(pop_size):
                    if not check_time() or (budget - calls_made) <= 0:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x_mean + sigma * (C @ z), -5, 5)
                    fy = safe_func(y)
                    offspring.append((fy, y))

                if len(offspring) < pop_size:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_local_f:
                    best_local_f = offspring[0][0]
                    best_local_x = offspring[0][1].copy()

                # Update mean
                w = np.zeros(pop_size)
                w[:min(pop_size // 2, 3)] = np.log(pop_size // 2 + 0.5) - np.log(np.arange(1, min(pop_size // 2 + 1, 4)))
                mu_eff = (w.sum() ** 2) / (w ** 2).sum()
                w = w / w.sum()

                x_new_mean = np.zeros(dim)
                for i in range(min(pop_size // 2, 3)):
                    x_new_mean += w[i] * offspring[i][1]

                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C @ (x_new_mean - x_mean))
                hsig = np.linalg.norm(ps) / chi_n * np.sqrt(1 - (1 - cs) ** (2 * (step + 1)))
                hsig = min(1.0, hsig)
                sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 5.0)

                # Update covariance
                pc = (1 - cc) * pc + cc * hsig * (x_new_mean - x_mean) / sigma
                C = (1 + cmu * hsig * (hsig - 1 + 2 * cmu) * (1 - cc) - cm1 - cm2) * C
                C += cmu * np.outer(pc, pc) + cm2 * (1 - hsig) * np.outer(ps, ps) / (sigma ** 2)

                # Symmetrize
                C = (C + C.T) / 2

                x_mean = x_new_mean

                # Convergence check
                if sigma < 1e-6:
                    break

            # Return best found by CMA-ES
            return best_local_x, best_local_f

        # Nelder-Mead local search
        def nelder_mead(x0, f0, max_steps=40):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (budget - calls_made) <= 0:
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.05 * (1 if xi[i] < 0 else -1)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, dim + 1):
                if not check_time() or (budget - calls_made) <= 0:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or (budget - calls_made) <= 0:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or (budget - calls_made) <= 0:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or (budget - calls_made) <= 0:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Phase 2: Multi-start (1+1)-ES from top random points
        top_k = min(4, len(points))
        for i in range(top_k):
            if not check_time() or (budget - calls_made) <= 15:
                break
            x_start = points[i][1].copy()
            f_start = points[i][0]
            remaining = budget - calls_made
            steps = min(50, remaining // 3)
            if steps > 0:
                one_plus_one_es(x_start, f_start, max_steps=steps)

        # Phase 3: CMA-ES refinement on current best
        remaining = budget - calls_made
        if remaining > 30 and best_x is not None and check_time():
            cma_steps = min(40, remaining // 2)
            if cma_steps > 5:
                cma_es(best_x, best_f, max_steps=cma_steps)

        # Phase 4: Final Nelder-Mead polish
        remaining = budget - calls_made
        if remaining > 15 and best_x is not None and check_time():
            nm_steps = min(35, remaining - 5)
            if nm_steps > 0:
                nelder_mead(best_x, best_f, max_steps=nm_steps)

        return best_x, best_f
# EVOLVE-BLOCK-END
