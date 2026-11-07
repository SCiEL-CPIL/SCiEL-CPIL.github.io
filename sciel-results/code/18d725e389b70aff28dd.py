import numpy as np
import time
import math

class Optimizer:
    """Hybrid DE + Nelder-Mead + CMA-ES optimizer for Katsuura function f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_evals, step=0.5):
            nonlocal best_f, best_x, evals_used
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step

            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if time.time() > deadline or evals_used >= budget:
                    return
                fvals[i] = evaluate(simplex[i])

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() > deadline or evals_used >= budget:
                                return
                            fvals[i] = evaluate(simplex[i])
                        break

        def cma_11(x0, max_evals, sigma0=0.3):
            nonlocal best_f, best_x, evals_used
            n = dim
            x = x0.copy()
            f = best_f
            sigma = sigma0
            C = np.eye(n)
            p_c = np.zeros(n)
            p_sigma = np.zeros(n)

            c_c = (4 + n) / (n + 4)
            c_sigma = 1 / (n + 3)
            c_1 = 2 / (n ** 2 + 18)
            c_mu = min(1 - c_1, 1 / (n + 2))
            chi_n = np.sqrt(n) * (1 - 1/(4*n))
            d_sigma = 1 + 2 * max(0, 1/np.sqrt(n+1) - 1) + c_sigma

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return
                if sigma < 1e-10:
                    break

                z = rng.standard_normal(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C += 1e-6 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C_sqrt = np.eye(n)

                y = C_sqrt @ z
                x_trial = x + sigma * y
                f_trial = evaluate(x_trial)

                if f_trial < f:
                    y_sigma = y / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * y_sigma
                    h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2/(n+1)) * chi_n
                    if h_sigma:
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                    y_c = C @ y / sigma
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * y_c

                    delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                    C = (1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.outer(y/sigma, y/sigma)
                    x = x_trial
                    f = f_trial

        # Phase 1: Structured diverse sampling
        x_center = np.zeros(dim)
        evaluate(x_center)

        # Vertices of [-4, 4]^dim
        for i in range(2 ** dim):
            if time.time() > deadline or evals_used >= budget:
                break
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 8 - 4
            evaluate(v)

        # Random points in [-4, 4]^dim
        n_random = 40
        for i in range(n_random):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 2: Differential Evolution
        pop_size = 40
        population = []
        pop_f = []

        # Seed population with best point and diverse random points
        population.append(best_x.copy())
        pop_f.append(best_f)

        for i in range(1, pop_size):
            if time.time() > deadline or evals_used >= budget:
                break
            x = best_x + rng.normal(0, 2.0, dim)
            x = clip(x)
            f = evaluate(x)
            population.append(x)
            pop_f.append(f)

        pop_f = np.array(pop_f)

        F = 0.7
        CR = 0.8
        stagnation_counter = 0
        max_stagnation = 150

        while time.time() < deadline and evals_used < budget:
            best_idx = np.argmin(pop_f)
            current_best = pop_f[best_idx]

            if current_best < best_f - 1e-10:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max_stagnation:
                for i in range(pop_size):
                    if time.time() > deadline or evals_used >= budget:
                        break
                    if i == 0:
                        population[i] = best_x.copy()
                        pop_f[i] = best_f
                    else:
                        population[i] = best_x + rng.normal(0, 2.5, dim)
                        population[i] = clip(population[i])
                        pop_f[i] = evaluate(population[i])
                stagnation_counter = 0
                continue

            for i in range(pop_size):
                if time.time() > deadline or evals_used >= budget:
                    break

                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                mutant = population[a] + F * (population[b] - population[c])

                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = clip(trial)
                f_trial = evaluate(trial)

                if f_trial <= pop_f[i]:
                    population[i] = trial
                    pop_f[i] = f_trial

        # Phase 3: Nelder-Mead refinement
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            nm_budget = min(100, remaining // 3)
            nelder_mead(best_x.copy(), nm_budget, step=0.5)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 4: CMA-ES (1+1) local refinement
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            es_budget = min(150, remaining // 2)
            cma_11(best_x.copy(), es_budget, sigma0=0.3)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 5: Final Nelder-Mead
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            nm_budget2 = min(80, remaining // 3)
            nelder_mead(best_x.copy(), nm_budget2, step=0.3)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 6: Quick restarts with local refinement
        remaining = budget - evals_used
        n_restarts = max(1, remaining // 40)
        for r in range(n_restarts):
            if time.time() > deadline or evals_used >= budget:
                break

            x_start = best_x + rng.normal(0, 0.8, dim)
            f_start = evaluate(x_start)

            if time.time() > deadline or evals_used >= budget:
                break

            nm_b = min(20, (budget - evals_used) // 2)
            if nm_b > 5:
                nelder_mead(x_start, nm_b, step=0.2)

        return best_x, best_f
