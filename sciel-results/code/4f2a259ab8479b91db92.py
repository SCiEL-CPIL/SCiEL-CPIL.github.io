# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive population size for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        state = {
            'calls': 0,
            'best_f': np.inf,
            'best_x': np.zeros(n)
        }

        def wrapped_func(x):
            if state['calls'] >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Initial population size (adaptive)
        pop_size = max(10, int(4 + 3 * np.log(n + 1)))
        min_pop_size = 4

        # CMA-ES parameters
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        L = np.eye(n)
        p_c = np.zeros(n)
        p_sigma = np.zeros(n)

        # Adaptation rates
        c_sigma = (n + 2) / (n + 5 + 2 * n / (n + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / n) / (4 + n + n)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * ((n + 1.) / (n + 4.)) ** 2 - 1)

        gen = 0
        max_gens = 50
        no_improve = 0
        best_f_prev = state['best_f']

        while state['calls'] < budget - 15 and gen < max_gens:
            if time.time() > deadline:
                break

            # Adaptive population size: shrink when converging
            if no_improve >= 5 and pop_size > min_pop_size:
                pop_size = max(min_pop_size, int(pop_size * 0.8))
            elif no_improve == 0 and pop_size < 10:
                pop_size = min(10, pop_size + 1)

            mu = max(2, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            offspring = []
            for i in range(pop_size):
                if state['calls'] >= budget - 15 or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_sample = x_mean + sigma * (L @ z)
                x_sample = np.clip(x_sample, -5, 5)
                fx = wrapped_func(x_sample)
                offspring.append((x_sample, fx))

            if len(offspring) < 2:
                break

            offspring.sort(key=lambda t: t[1])

            if offspring[0][1] < state['best_f']:
                state['best_f'] = offspring[0][1]
                state['best_x'] = offspring[0][0].copy()

            # Track improvement
            if state['best_f'] < best_f_prev - 1e-10:
                no_improve = 0
                best_f_prev = state['best_f']
            else:
                no_improve += 1

            parents = [offspring[i][0] for i in range(min(mu, len(offspring)))]
            actual_mu = len(parents)
            if actual_mu < 2:
                break

            # Recompute weights for actual mu
            w_actual = np.log(actual_mu + 0.5) - np.log(np.arange(1, actual_mu + 1))
            w_actual /= w_actual.sum()
            mu_eff_actual = 1.0 / np.sum(w_actual ** 2)

            new_mean = np.sum([w_actual[i] * parents[i] for i in range(actual_mu)], axis=0)

            y = (new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff_actual) * y

            rank_one = np.outer(y, y)
            rank_mu = np.zeros((n, n))
            for i in range(actual_mu):
                y_i = (parents[i] - x_mean) / sigma
                rank_mu += w_actual[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * (c_mu - c_1) / c_1 - c_1) * C + c_1 * (rank_one + rank_mu)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)

            # Sigma update
            h_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff_actual) * (L.T @ (L @ y))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff_actual) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            x_mean = new_mean
            gen += 1

            if sigma < 1e-4:
                break

        # Phase 2: Local refinement with coordinate descent + Nelder-Mead
        if state['calls'] < budget - 10 and time.time() < deadline:
            self._local_refine(wrapped_func, state['best_x'].copy(), state, deadline)

        return state['best_x'], state['best_f']

    def _local_refine(self, wrapped_func, start_x, state, deadline):
        """Coordinate descent followed by Nelder-Mead."""
        n = self.dim
        budget = self.budget
        x = start_x.copy()
        f_x = state['best_f']

        # Coordinate descent for quick initial improvement
        step = 0.5
        for _ in range(30):
            if state['calls'] >= budget - 10 or time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break
                for direction in [1, -1]:
                    x_try = x.copy()
                    x_try[i] = x[i] + direction * step
                    x_try = np.clip(x_try, -5, 5)
                    f_try = wrapped_func(x_try)
                    if f_try < f_x - 1e-12:
                        x = x_try
                        f_x = f_try
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break

        # Nelder-Mead for final refinement
        if state['calls'] >= budget - 10 or time.time() > deadline:
            return

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        simplex = [x.copy()]
        for i in range(n):
            pt = x.copy()
            step = 0.1
            if pt[i] + step > 5:
                pt[i] = pt[i] - step
            elif pt[i] - step < -5:
                pt[i] = pt[i] + step
            else:
                pt[i] = pt[i] + step
            simplex.append(pt)

        f_values = []
        for pt in simplex:
            if state['calls'] >= budget - 10 or time.time() > deadline:
                return
            f_values.append(wrapped_func(pt))

        if len(f_values) < n + 1:
            return

        iter_count = 0
        max_iter = 40

        while state['calls'] < budget - 10 and iter_count < max_iter and time.time() < deadline:
            sorted_indices = np.argsort(f_values)
            simplex = [simplex[i] for i in sorted_indices]
            f_values = [f_values[i] for i in sorted_indices]

            if f_values[0] < state['best_f']:
                state['best_f'] = f_values[0]
                state['best_x'] = simplex[0].copy()

            if state['calls'] >= budget - 10 or time.time() > deadline:
                break

            x_worst = simplex[-1]
            f_worst = f_values[-1]
            x_best = simplex[0]

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = wrapped_func(x_r)

            if state['calls'] >= budget - 10 or time.time() > deadline:
                break

            if f_r < f_values[0]:
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = wrapped_func(x_e)

                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (x_r - centroid)

                x_c = np.clip(x_c, -5, 5)
                f_c = wrapped_func(x_c)

                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break

                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if state['calls'] >= budget - 10 or time.time() > deadline:
                            break
                        f_values[i] = wrapped_func(simplex[i])

                    if state['calls'] >= budget - 10 or time.time() > deadline:
                        break
                    simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1)])
                    if simplex_size < 1e-6:
                        break

            iter_count += 1
# EVOLVE-BLOCK-END
