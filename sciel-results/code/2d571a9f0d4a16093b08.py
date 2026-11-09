"""Differential Evolution with CMA-ES polishing for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Differential Evolution (jDE) with CMA-ES polishing for multimodal functions.
    DE provides robust global exploration; CMA-ES refines the best solution locally.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        rng = np.random.default_rng(42)

        # --- CMA-ES Implementation (for polishing) ---
        def cma_es(x0, sigma0, pop_size, max_calls, sigma_min=1e-10, sigma_max=10.0):
            nonlocal n_calls, best_x, best_f

            x = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            N = dim

            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (N + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
            c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
            c_1 = 2 / ((N + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
            chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

            pc = np.zeros(N)
            ps = np.zeros(N)
            C = np.eye(N)

            best_local_x = x.copy()
            best_local_f = np.inf

            for gen in range(max_calls // pop_size + 1):
                if n_calls >= budget or check_time():
                    break

                population = []
                fitnesses = []
                for i in range(pop_size):
                    if n_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(N)
                    y = x + sigma * z
                    fy, y_eval = call_func(y)
                    if fy is None:
                        continue
                    population.append(y_eval)
                    fitnesses.append(fy)

                if len(population) < 2:
                    break

                order = np.argsort(fitnesses)
                population = [population[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]

                for i in range(min(mu, len(population))):
                    if fitnesses[i] < best_local_f:
                        best_local_f = fitnesses[i]
                        best_local_x = population[i].copy()
                        if fitnesses[i] < best_f:
                            best_f = fitnesses[i]
                            best_x = population[i].copy()

                x_new = np.zeros(N)
                for i in range(mu):
                    x_new += weights[i] * population[i]
                x = x_new

                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(N)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_N * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (n_calls + 1)))) < (1.4 + 2 / (N + 1)) * chi_N
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

                delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1)) * c_sigma
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((N, N))
                for i in range(mu):
                    zi = (population[i] - x_new) / sigma if sigma > 0 else np.zeros(N)
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * (c_c * (2 - c_c) * np.eye(N) - C)) + c_mu * rank_mu
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(N)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
                sigma = max(sigma_min, min(sigma_max, sigma))

                if sigma < sigma_min or np.all(np.abs(C) < 1e-10):
                    break

            return best_local_x, best_local_f

        # --- Differential Evolution (jDE) ---
        def differential_evolution(max_calls):
            nonlocal n_calls, best_x, best_f

            # Population size
            pop_size = max(10, 4 * dim)
            pop_size = min(pop_size, budget - n_calls - 50)  # Leave room for CMA-ES
            if pop_size < 4:
                pop_size = 4

            # Initialize population
            pop = []
            for i in range(pop_size):
                x = rng.uniform(-5, 5, dim)
                fx, x_eval = call_func(x)
                if fx is None:
                    break
                pop.append((x_eval, fx))

            if not pop:
                return

            # Sort
            pop.sort(key=lambda t: t[1])

            # Update best
            for x, f in pop:
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # jDE parameters
            F = 0.5
            CR = 0.1
            F_min, F_max = 0.1, 1.0
            CR_min, CR_max = 0.1, 0.9

            # Run DE
            while n_calls < max_calls and not check_time():
                # Adaptive parameters
                F = F_min + rng.random() * (F_max - F_min)
                CR = CR_min + rng.random() * (CR_max - CR_min)

                # New population
                new_pop = []
                for i in range(pop_size):
                    if n_calls >= max_calls or check_time():
                        break

                    # Select 3 distinct individuals
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(idxs, 3, replace=False)
                    x_a, _ = pop[a]
                    x_b, _ = pop[b]
                    x_c, _ = pop[c]
                    x_target, f_target = pop[i]

                    # Mutation
                    v = x_target + F * (x_a - x_b) + F * (rng.random() < CR) * (x_c - x_target)
                    v = np.clip(v, -5, 5)

                    # Evaluate
                    fv, v_eval = call_func(v)
                    if fv is None:
                        continue

                    # Selection
                    if fv <= f_target:
                        new_pop.append((v_eval, fv))
                    else:
                        new_pop.append((x_target, f_target))

                # Sort new population
                new_pop.sort(key=lambda t: t[1])
                pop = new_pop

                # Update best
                for x, f in pop:
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

            return pop

        # --- Main Strategy ---
        # Phase 1: Differential Evolution for global exploration
        de_budget = int(budget * 0.6)  # 60% of budget for DE
        differential_evolution(de_budget)

        # Phase 2: CMA-ES polishing on the best point
        if n_calls < budget and not check_time() and best_x is not None:
            remaining = budget - n_calls
            cma_budget = min(remaining, 150)
            if cma_budget > 10:
                pop_size = max(6, 4 + int(3 * np.log(dim)))
                sigma0 = 0.5  # Small initial sigma for polishing
                cma_es(best_x, sigma0, pop_size, cma_budget)

        # Phase 3: Final local refinement with (1+1)-ES if budget remains
        if n_calls < budget and not check_time() and best_x is not None:
            remaining = budget - n_calls
            if remaining > 5:
                x = best_x.copy()
                f = best_f
                sigma = 0.1
                for _ in range(remaining):
                    if n_calls >= budget or check_time():
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy, y_eval = call_func(y)
                    if fy is None:
                        break
                    if fy < f:
                        x = y_eval
                        f = fy
                        sigma *= 1.1
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        sigma *= 0.8
                        if sigma < 1e-6:
                            break

        if best_x is None:
            return None

        return best_x, best_f

# EVOLVE-BLOCK-END
