"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton-seed multi-start CMA-ES for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or not time_ok():
                return float('inf')
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else float('inf')

        # --- Phase 1: Deterministic Halton sequence for early global coverage ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            point = np.zeros(dim)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                point[d] = halton(n, primes[d % len(primes)])
            return point

        # Generate Halton points for initial exploration
        n_halton = min(40, budget // 10)
        halton_points = []
        for i in range(1, n_halton + 1):
            pt = halton_point(i, dim)
            x = pt * 10.0 - 5.0
            fx = safe_eval(x)
            halton_points.append((fx, x.copy()))

        if best_x is None:
            best_x = np.zeros(dim)
            safe_eval(best_x)

        # Sort Halton points by fitness
        halton_points.sort(key=lambda t: t[0])

        # --- Phase 2: Multi-start CMA-ES refinement from best diverse Halton points ---
        # Select top diverse points for refinement
        selected_starts = []
        min_dist = 1.0
        for fx, x in halton_points:
            if not time_ok() or calls_made >= budget - 5:
                break
            too_close = False
            for _, sx in selected_starts:
                if np.linalg.norm(x - sx) < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append((fx, x))
            if len(selected_starts) >= 6:
                break

        if len(selected_starts) < 4:
            for fx, x in halton_points:
                if len(selected_starts) >= 4:
                    break
                if (fx, x) not in selected_starts:
                    selected_starts.append((fx, x))

        # CMA-ES implementation for each start
        def cma_es(x0, max_calls, sigma_init=1.5):
            nonlocal calls_made, best_f, best_x

            if not time_ok() or calls_made >= budget - 3:
                return

            x = x0.copy()
            sigma = sigma_init
            lambda_ = 8
            mu = 4
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            c1 = 1 / (dim + 3)
            cmu = 2 / ((dim + 2) ** 2 + 5)
            cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
            cs = (dim + 1.7) / (dim + 9)
            damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            step_size = sigma

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            for gen in range(200):
                if calls_made >= budget or not time_ok():
                    break

                offspring = []
                for i in range(lambda_):
                    if calls_made >= budget or not time_ok():
                        break
                    z = np.random.standard_normal(dim)
                    y = L @ z
                    xi = x + step_size * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_made += 1
                    if np.isfinite(fi):
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()
                        offspring.append((fi, xi))

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])

                mu_actual = min(mu, len(offspring))
                selected = [offspring[i][1] for i in range(mu_actual)]

                new_x = np.zeros(dim)
                for i in range(mu_actual):
                    new_x += weights[i] * selected[i]

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
                step_size *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                step_size = np.clip(step_size, 1e-10, 50)

                v = (new_x - x) / step_size
                rank_one = c1 * (np.outer(v, v) + (1 - hsig) * cc * (2 - cc) * np.outer(pc, pc))
                rank_mu = cmu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) / step_size ** 2 for i in range(mu_actual))
                C = (1 - c1 - cmu) * C + rank_one + rank_mu

                pc = (1 - cc) * pc + cc * hsig * v

                x = new_x

                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

                if step_size < 1e-6:
                    break

        # Run CMA-ES from each selected start with limited budget per start
        n_starts = len(selected_starts)
        calls_per_start = max(30, (budget - calls_made) // max(1, n_starts))

        for fx0, x0 in selected_starts:
            if not time_ok() or calls_made >= budget - 3:
                break
            cma_es(x0.copy(), calls_per_start)

        # --- Phase 3: Final local polish with Nelder-Mead ---
        from scipy.optimize import minimize

        if calls_made < budget - 2 and time_ok() and best_x is not None:
            try:
                res = minimize(
                    safe_eval,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
