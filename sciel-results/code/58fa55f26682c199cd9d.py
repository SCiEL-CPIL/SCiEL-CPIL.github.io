# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES + Powell + Nelder-Mead for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
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

        # Phase 1: CMA-ES
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

        while state['calls'] < budget - 40 and gen < max_gens:
            if time.time() > deadline:
                break

            offspring = []
            for i in range(pop_size):
                if state['calls'] >= budget - 40 or time.time() > deadline:
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

        # Phase 2: Powell method (direction-set optimization)
        if state['calls'] < budget - 20:
            self._powell_refine(wrapped_func, state['best_x'].copy(), state, deadline)

        # Phase 3: Nelder-Mead refinement
        if state['calls'] < budget - 5:
            self._nelder_mead_refine(wrapped_func, state['best_x'].copy(), state, deadline)

        return state['best_x'], state['best_f']

    def _powell_refine(self, wrapped_func, start_x, state, deadline):
        """Powell's method: direction-set optimization with adaptive direction updates."""
        n = self.dim
        budget = self.budget
        x = start_x.copy()
        f_x = state['best_f']
        f0 = f_x

        # Initial directions: standard basis
        directions = np.eye(n)

        max_iter = 30
        for _ in range(max_iter):
            if state['calls'] >= budget - 10 or time.time() > deadline:
                break

            f_prev = f_x
            shift = np.zeros(n)

            for i in range(n):
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break

                # Minimize along directions[i]
                d = directions[i]
                # Find step size via golden section / bisection
                # First, determine a range
                a, b = -1.0, 1.0
                fa = f_x
                fb = wrapped_func(x + b * d)

                # If fb is better, expand in that direction
                if fb < fa:
                    for _ in range(10):
                        if state['calls'] >= budget - 10 or time.time() > deadline:
                            break
                        a = b
                        fa = fb
                        b *= 2.0
                        fb = wrapped_func(x + b * d)
                        if fb >= fa:
                            break
                else:
                    # Expand in negative direction
                    a, b = b, a
                    fa, fb = fb, fa
                    for _ in range(10):
                        if state['calls'] >= budget - 10 or time.time() > deadline:
                            break
                        b = a
                        fb = fa
                        a *= 2.0
                        fa = wrapped_func(x + a * d)
                        if fa >= fb:
                            break

                # Golden section search in [a, b]
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break

                golden = (np.sqrt(5) - 1) / 2
                lo, hi = a, b
                x1 = hi - golden * (hi - lo)
                x2 = lo + golden * (hi - lo)
                f1 = wrapped_func(x + x1 * d)
                f2 = wrapped_func(x + x2 * d)

                for _ in range(20):
                    if state['calls'] >= budget - 10 or time.time() > deadline:
                        break
                    if f1 < f2:
                        hi = x2
                        x2 = x1
                        f2 = f1
                        x1 = hi - golden * (hi - lo)
                        f1 = wrapped_func(x + x1 * d)
                    else:
                        lo = x1
                        x1 = x2
                        f1 = f2
                        x2 = lo + golden * (hi - lo)
                        f2 = wrapped_func(x + x2 * d)

                # Best step
                step = min(x1, x2, key=lambda t: wrapped_func(x + t * d))
                f_new = wrapped_func(x + step * d)
                if f_new < f_x:
                    x = x + step * d
                    f_x = f_new
                    shift = shift + step * d
                    if f_x < state['best_f']:
                        state['best_f'] = f_x
                        state['best_x'] = x.copy()

            # Update directions
            if f_x < f_prev - 1e-10:
                # Replace the direction with the largest change
                changes = np.abs(shift)
                idx = np.argmax(changes)
                directions = np.roll(directions, -1, axis=1)
                directions[:, -1] = shift
            else:
                break

    def _nelder_mead_refine(self, wrapped_func, start_x, state, deadline):
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
            if state['calls'] >= budget - 5:
                return
            f_values.append(wrapped_func(pt))

        if len(f_values) < n + 1:
            return

        iter_count = 0
        max_iter = 60

        while state['calls'] < budget - 5 and iter_count < max_iter:
            if time.time() > deadline:
                break

            sorted_indices = np.argsort(f_values)
            simplex = [simplex[i] for i in sorted_indices]
            f_values = [f_values[i] for i in sorted_indices]

            if f_values[0] < state['best_f']:
                state['best_f'] = f_values[0]
                state['best_x'] = simplex[0].copy()

            if state['calls'] >= budget - 5:
                break

            x_worst = simplex[-1]
            f_worst = f_values[-1]
            x_best = simplex[0]

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = wrapped_func(x_r)

            if state['calls'] >= budget - 5:
                break

            if f_r < f_values[0]:
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
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
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
                    shrink = True
                    for i in range(1, n + 1):
                        if state['calls'] >= budget - 5:
                            shrink = False
                            break
                        simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = wrapped_func(simplex[i])
                    if not shrink:
                        break
            iter_count += 1
# EVOLVE-BLOCK-END
