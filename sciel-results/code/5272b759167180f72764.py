"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with adaptive (1+lambda)-ES restarts and coordinate polish for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for sign in (1, -1):
                x = np.zeros(d)
                x[i] = sign * 2.5
                points.append(x)
        for _ in range(8):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: CMA-ES with proper covariance adaptation
        def cma_es(x0, sigma0, max_calls, rng_local):
            """Run CMA-ES starting from x0 with initial sigma."""
            nonlocal best_x, best_fx, calls_made

            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            c1 = (2 + 2 * mueff / n) / ((n + 2) ** 2 + 2 * mueff / n)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / n))

            C = np.eye(n)
            C_eigval = np.ones(n)
            C_eigvec = np.eye(n)
            C_eigdec = False

            xmean = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            local_best_x = x0.copy()
            local_best_fx = np.inf

            def C_eig_decomp():
                nonlocal C_eigval, C_eigvec, C_eigdec
                if not C_eigdec:
                    C_eigval, C_eigvec = np.linalg.eigh(C)
                    C_eigval = np.maximum(C_eigval, 1e-20)
                    C_eigdec = True
                return C_eigval, C_eigvec

            CS = None
            if C_eigdec:
                CS = C_eigvec @ np.diag(np.sqrt(C_eigval))
            else:
                CS = np.eye(n)

            while calls_made < max_calls and time.time() < deadline:
                if not time_ok():
                    break

                ys = []
                for _ in range(lam):
                    z = rng_local.standard_normal(n)
                    y = xmean + sigma * (CS @ z)
                    ys.append(y)

                fvals = []
                for y in ys:
                    if not time_ok():
                        break
                    y_clipped = np.clip(y, -5, 5)
                    f = call_func(y_clipped)
                    fvals.append(f)

                if len(fvals) < lam:
                    break

                if not time_ok():
                    break

                indices = np.argsort(fvals)
                ys_sorted = [ys[i] for i in indices]
                fvals_sorted = [fvals[i] for i in indices]

                if fvals_sorted[0] < local_best_fx:
                    local_best_fx = fvals_sorted[0]
                    local_best_x = ys_sorted[0].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best_x.copy()

                y_w = np.array(ys_sorted[:mu])
                xmean_new = np.sum(w[:, None] * y_w, axis=0)

                y_sigma = (xmean_new - xmean) / (sigma * chi_n)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma

                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** 2)) < (1.4 + 2 / (n + 1)) * chi_n
                hs = float(hsig)

                p_c = (1 - cs) * p_c + hs * np.sqrt(cs * (2 - cs) * mueff) * (xmean_new - xmean) / sigma

                C_eig_decomp()
                C_old = C_eigvec @ np.diag(C_eigval) @ C_eigvec.T

                delta_p = p_c / sigma
                rank1_update = c1 * (np.outer(delta_p, delta_p) + (1 - hs) * c1 * (1 - c1) * C_old)

                rankmu_update = np.zeros((n, n))
                for i in range(mu):
                    y_i = (y_w[i] - xmean) / sigma
                    rankmu_update += w[i] * np.outer(y_i, y_i)
                rankmu_update *= cmu

                C = (1 - c1 - cmu) * C_old + rank1_update + rankmu_update
                C = 0.5 * (C + C.T)

                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.maximum(C_eigval, 1e-20)
                C_eigdec = True

                CS = C_eigvec @ np.diag(np.sqrt(C_eigval))

                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                xmean = xmean_new

                if sigma < 1e-12 or np.all(np.abs(xmean) > 4.9):
                    break

            return local_best_x, local_best_fx

        # Phase 3: Adaptive (1+lambda)-ES with restarts
        def es_restart(x0, fx0, max_calls, rng_local):
            """(1+lambda)-ES with adaptive sigma and restarts on stagnation."""
            nonlocal best_x, best_fx, calls_made

            lambda_ = 4
            sigma = 0.5
            x = np.clip(x0.copy(), -5, 5)
            fx = fx0
            no_improve = 0
            max_no_improve = 30

            while calls_made < max_calls and time.time() < deadline:
                if not time_ok():
                    break

                candidates = []
                for _ in range(lambda_):
                    y = x + sigma * rng_local.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    candidates.append(y)

                cands_fx = []
                for y in candidates:
                    if not time_ok():
                        break
                    fy = call_func(y)
                    cands_fx.append(fy)

                if len(cands_fx) < lambda_:
                    break

                best_idx = np.argmin(cands_fx)
                best_cand_fx = cands_fx[best_idx]
                best_cand = candidates[best_idx]

                if best_cand_fx < fx:
                    x = best_cand
                    fx = best_cand_fx
                    no_improve = 0
                    sigma = min(sigma * 1.2, 2.0)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.85, 0.01)

                if no_improve >= max_no_improve and calls_made < max_calls - 20:
                    no_improve = 0
                    sigma = 1.0
                    x = rng_local.uniform(-4, 4, d)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

            return x, fx

        # Phase 4: Coordinate descent polish
        def coord_descent(x0, max_calls, rng_local):
            """Coordinate descent with adaptive step size for final polish."""
            nonlocal best_x, best_fx, calls_made

            x = np.clip(x0.copy(), -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            steps = np.ones(d) * 0.2
            min_step = 1e-7
            max_iter = max(1, max_calls - calls_made)

            for _ in range(max_iter):
                if not time_ok():
                    break
                improved = False
                for i in range(d):
                    if not time_ok():
                        break
                    x_trial = x.copy()
                    x_trial[i] = np.clip(x[i] + steps[i], -5, 5)
                    fx_trial = call_func(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        steps[i] = min(steps[i] * 1.2, 0.5)
                        improved = True
                    else:
                        x_trial = x.copy()
                        x_trial[i] = np.clip(x[i] - steps[i], -5, 5)
                        fx_trial = call_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                            steps[i] = min(steps[i] * 1.2, 0.5)
                            improved = True
                        else:
                            steps[i] = max(steps[i] * 0.5, min_step)

                if not improved and np.all(steps < min_step * 5):
                    break

            return best_x, best_fx

        # Main optimization loop: alternate CMA-ES and (1+lambda)-ES
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made

            # First CMA-ES run: 40% of remaining
            cma1_calls = min(remaining, int(remaining * 0.4))
            _, _ = cma_es(best_x.copy(), 1.5, calls_made + cma1_calls, rng)

        # (1+lambda)-ES refinement
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                es_calls = min(remaining, int(remaining * 0.35))
                _, _ = es_restart(best_x.copy(), best_fx, calls_made + es_calls, rng)

        # Second CMA-ES run with smaller sigma
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 40:
                cma2_calls = min(remaining, int(remaining * 0.3))
                _, _ = cma_es(best_x.copy(), 0.3, calls_made + cma2_calls, rng)

        # Coordinate descent polish
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                cd_calls = min(remaining, 40)
                _, _ = coord_descent(best_x.copy(), calls_made + cd_calls, rng)

        # Final high-precision polish
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(20, budget - calls_made)
            step = 0.01
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + step * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    step = max(step * 0.7, 1e-5)
                else:
                    step = max(step * 0.8, 1e-5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
