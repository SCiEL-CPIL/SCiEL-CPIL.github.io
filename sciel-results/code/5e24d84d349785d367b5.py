import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        history_best = []

        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def record_best():
            history_best.append(best_f)

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        # Phase 1: Differential Evolution for global exploration
        rng = np.random.default_rng(42)
        pop_size = 20
        F = 0.8
        CR = 0.9
        de_generations = 8

        # Initialize population
        pop = [best_x.copy()]
        for _ in range(pop_size - 1):
            pop.append(rng.uniform(-5, 5, dim))

        # Evaluate initial population
        for i in range(pop_size):
            if eval_count >= budget or time_check():
                break
            wrapped_func(pop[i])
        record_best()

        # DE loop
        for gen in range(de_generations):
            if eval_count >= budget or time_check():
                break

            new_pop = []
            for i in range(pop_size):
                if eval_count >= budget or time_check():
                    break

                # Select three distinct random indices
                a, b, c = rng.choice(pop_size, 3, replace=False)
                while a == b or a == c or b == c:
                    a, b, c = rng.choice(pop_size, 3, replace=False)

                # Create trial vector
                trial = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = pop[a][j] + F * (pop[b][j] - pop[c][j])
                    else:
                        trial[j] = pop[a][j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                trial_f = wrapped_func(trial)
                record_best()

                # Selection
                if trial_f <= wrapped_func(pop[i]):
                    new_pop.append(trial)
                else:
                    new_pop.append(pop[i].copy())

                if eval_count >= budget or time_check():
                    break

            if len(new_pop) < pop_size:
                # Pad with existing population
                for i in range(len(new_pop), pop_size):
                    new_pop.append(pop[i].copy())
            pop = new_pop

        # Phase 2: CMA-ES for local refinement
        if eval_count < budget and not time_check():
            lam = 10
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))

            x_mean = best_x.copy()
            sigma = 0.5
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
            c_hs = 1.1 + 2.0 / (dim + 1.7)
            generation = 0

            while eval_count < budget and not time_check():
                ys = [B @ (D * np.random.standard_normal(dim)) for _ in range(lam)]
                xs = x_mean + sigma * np.array(ys)

                num_evals = min(lam, budget - eval_count)
                if num_evals <= 0:
                    break

                fitnesses = []
                eval_xs = []
                for i in range(num_evals):
                    f = wrapped_func(xs[i])
                    fitnesses.append(f)
                    eval_xs.append(xs[i])
                    record_best()
                    if eval_count >= budget:
                        break

                if num_evals < lam:
                    break

                fitnesses = np.array(fitnesses)
                order = np.argsort(fitnesses)
                xs_sorted = np.array(eval_xs)[order]
                ys_sorted = np.array(ys)[order][:mu]

                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                y_mean = (new_mean - x_mean) / sigma

                norm_y_mean = np.linalg.norm(y_mean)
                sqrt_trace_C = np.sqrt(np.trace(C))
                ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0

                p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
                norm_p_sigma = np.linalg.norm(p_sigma)
                gen = generation + 1
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1.4 + 2 / (dim + 1)) * chi_n)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                generation = gen

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                rank_one = np.outer(p_c, p_c)
                if not h_sigma:
                    rank_one += c_c * (2 - c_c) * C
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])

                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                x_mean = new_mean

                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.clip(D2, 1e-10, None)
                    D = np.sqrt(D2)
                except:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)

                if sigma < 1e-5:
                    break

        # Phase 3: Coordinate descent polish
        if eval_count < budget and not time_check():
            local_x = best_x.copy()
            local_f = best_f
            step = 0.05
            max_local_evals = min(50, budget - eval_count)
            for _ in range(max_local_evals):
                if eval_count >= budget or time_check():
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        trial_x = local_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, -5, 5)
                        if eval_count >= budget or time_check():
                            break
                        trial_f = wrapped_func(trial_x)
                        record_best()
                        if trial_f < local_f:
                            local_x = trial_x
                            local_f = trial_f
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break

        # Fill history if needed
        while len(history_best) < budget:
            history_best.append(history_best[-1] if history_best else best_f)

        return best_x, best_f
