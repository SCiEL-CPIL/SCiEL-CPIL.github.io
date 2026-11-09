# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead optimizer for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        # Shared mutable state
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

        # =========================================================
        # Phase 1: CMA-ES
        # =========================================================
        pop_size = 10
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        p_c = np.zeros(n)
        p_sigma = np.zeros(n)

        c_sigma = (n + 2) / (n + 5 + 2 * n / (n + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / n) / (4 + n + n)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * ((n + 1.) / (n + 4.)) ** 2 - 1)

        gen = 0
        max_gens = 40

        while state['calls'] < budget - 20 and gen < max_gens:
            if time.time() > deadline:
                break

            offspring = []
            for i in range(pop_size):
                if state['calls'] >= budget - 20 or time.time() > deadline:
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

            parents = [offspring[i][0] for i in range(mu)]
            new_mean = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)

            y = (new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            rank_one = np.outer(y, y)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (parents[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * (c_mu - c_1) / c_1 - c_1) * C + c_1 * (rank_one + rank_mu)

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)

            h_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (L @ y))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            x_mean = new_mean
            gen += 1

            if sigma < 1e-4:
                break

        # =========================================================
        # Phase 2: Nelder-Mead refinement
        # =========================================================
        if state['calls'] < budget - 10:
            self._nelder_mead_refine(wrapped_func, state['best_x'].copy(), state, deadline)

        # =========================================================
        # Phase 3: Fine Nelder-Mead polish from best point
        # =========================================================
        if state['calls'] < budget - 5:
            self._nelder_mead_refine(wrapped_func, state['best_x'].copy(), state, deadline, step=0.01, max_iter=30)

        return state['best_x'], state['best_f']

    def _nelder_mead_refine(self, wrapped_func, start_x, state, deadline, step=0.1, max_iter=50):
        """Nelder-Mead refinement using shared state."""
        n = self.dim
        budget = self.budget

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        simplex = [start_x.copy()]
        for i in range(n):
            pt = start_x.copy()
            if pt[i] + step > 5:
                pt[i] = pt[i] - step
            elif pt[i] - step < -5:
                pt[i] = pt[i] + step
            else:
                pt[i] = pt[i] + step
            simplex.append(pt)

        f_values = []
        for pt in simplex:
            if state['calls'] >= budget - 5 or time.time() > deadline:
                return
            f_values.append(wrapped_func(pt))

        if len(f_values) < n + 1:
            return

        iter_count = 0
        stall_count = 0
        prev_best = state['best_f']

        while state['calls'] < budget - 5 and iter_count < max_iter and time.time() < deadline:
            # Sort
            sorted_indices = np.argsort(f_values)
            simplex = [simplex[i] for i in sorted_indices]
            f_values = [f_values[i] for i in sorted_indices]

            if f_values[0] < state['best_f']:
                state['best_f'] = f_values[0]
                state['best_x'] = simplex[0].copy()

            # Stall detection
            if f_values[0] < prev_best - 1e-12:
                prev_best = f_values[0]
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 15:
                    break

            if state['calls'] >= budget - 5:
                break

            x_worst = simplex[-1]
            f_worst = f_values[-1]
            x_best = simplex[0]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = wrapped_func(x_r)

            if state['calls'] >= budget - 5:
                break

            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = wrapped_func(x_e)

                if state['calls'] >= budget - 5:
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Simple reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (x_r - centroid)

                x_c = np.clip(x_c, -5, 5)
                f_c = wrapped_func(x_c)

                if state['calls'] >= budget - 5:
                    break

                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = wrapped_func(simplex[i])

                        if state['calls'] >= budget - 5:
                            break

                    if state['calls'] >= budget - 5:
                        break

                    # Check if simplex is too small
                    simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1)])
                    if simplex_size < 1e-8:
                        break

            iter_count += 1
# EVOLVE-BLOCK-END
