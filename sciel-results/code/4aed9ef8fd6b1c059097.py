"""Hybrid Bayesian Optimization + CMA-ES for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Hybrid BO + CMA-ES for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_arr = np.asarray(x, dtype=np.float64)
                x_clipped = np.clip(x_arr, -5.0, 5.0)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()

        def gp_predict(X_train, Y_train, X_new, length_scale=1.5):
            n = X_train.shape[0]
            m = X_new.shape[0]
            X_tr = (X_train - np.array([-5.0]*dim)) / 10.0
            X_ne = (X_new - np.array([-5.0]*dim)) / 10.0
            def rbf(A, B, ls):
                sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                return np.exp(-0.5 * np.maximum(sq, 0) / (ls*ls))
            K = rbf(X_tr, X_tr, length_scale) + 1e-6 * np.eye(n)
            Ks = rbf(X_tr, X_ne, length_scale)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, Ks.T)
                mean = v.T @ alpha
                std = np.sqrt(np.maximum(1.0 - np.sum(v**2, axis=0), 0))
            except np.linalg.LinAlgError:
                mean = np.full(m, Y_train.mean())
                std = np.full(m, Y_train.std() + 1e-3)
            return mean, std

        def ei(mean, std, bf, xi=0.005):
            z = (bf - mean - xi) / (std + 1e-10)
            return np.maximum((bf - mean - xi) * norm.cdf(z) + std * norm.pdf(z), 0)

        def bayesian_optimize(x_init, f_init, n_bo_calls, rng):
            nonlocal best_x, best_f, n_calls
            X_train = [x_init.copy()]
            Y_train = [f_init]
            n_extra = min(5, n_bo_calls // 5)
            for _ in range(n_extra):
                if n_calls >= budget or check_time():
                    break
                x_try = rng.uniform(-5, 5, dim)
                f_try, x_eval = call_func(x_try)
                if f_try is None:
                    continue
                X_train.append(x_eval.copy())
                Y_train.append(f_try)
                update(x_eval, f_try)
            for _ in range(n_bo_calls - n_extra):
                if n_calls >= budget or check_time():
                    break
                X_cand = rng.uniform(-5, 5, (200, dim))
                mean, std = gp_predict(np.array(X_train), np.array(Y_train), X_cand)
                eival = ei(mean, std, best_f)
                idx = np.argmax(eival)
                f_next, x_eval = call_func(X_cand[idx])
                if f_next is None:
                    continue
                X_train.append(x_eval.copy())
                Y_train.append(f_next)
                update(x_eval, f_next)
            return best_x, best_f

        def cma_es(x0, sigma0, max_calls, rng):
            n = dim
            lam = max(6, 4 + int(3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = (2.0 / ((n + 1.3) ** 2 + mueff)) * (1.0 - cc)
            cmu = (2.0 * (mueff / n - 1.0) / ((n + 2) ** 2 + mueff)) * (min(1.0, c1) - c1)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
            x = np.asarray(x0, dtype=np.float64).copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf
            gen = 0
            stagnation = 0

            def sample_pop(mean, sig, C_mat):
                pop = np.zeros((lam, n))
                try:
                    ev, evec = np.linalg.eigh(C_mat)
                    ev = np.clip(ev, 1e-20, None)
                    Cinvsqrt = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
                except np.linalg.LinAlgError:
                    Cinvsqrt = np.eye(n)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = mean + sig * (Cinvsqrt @ z)
                return pop, Cinvsqrt

            pop, Cinvsqrt = sample_pop(x, sigma, C)
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    break
                res = call_func(pop[i])
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                f, x_eval = res
                calls_used += 1
                fvals[i] = f
                pop[i] = x_eval
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                if fvals[0] < local_best_f - 1e-10:
                    stagnation = 0
                else:
                    stagnation += 1
                if stagnation > 5 and gen > 3:
                    break
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pop[i]
                x = x_new
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinvsqrt @ (x - x_new)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * Cinvsqrt @ (x - x_new)
                C = (1.0 - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) + (1.0 + (1.0 - cc) * cc) * C - C)
                for i in range(mu):
                    diff = pop[i] - x
                    C += cmu * weights[i] * np.outer(diff, diff)
                C = (C + C.T) / 2.0
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)
                if sigma < 1e-10:
                    break
                pop, Cinvsqrt = sample_pop(x, sigma, C)
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    res = call_func(pop[i])
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f, x_eval = res
                    calls_used += 1
                    fvals[i] = f
                    pop[i] = x_eval
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                gen += 1
            return local_best_x, local_best_f, calls_used

        def local_polish(x0, max_calls, rng):
            x = np.asarray(x0, dtype=np.float64).copy()
            res = call_func(x)
            if res[0] is None:
                return x, np.inf
            f, x_eval = res
            best_local_f = f
            best_local_x = x_eval.copy()
            step = 0.5
            for _ in range(max_calls):
                if check_time() or n_calls >= budget:
                    break
                improved = False
                for d in range(dim):
                    if check_time() or n_calls >= budget:
                        break
                    for sign in (1.0, -1.0):
                        if n_calls >= budget or check_time():
                            break
                        trial = x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5.0, 5.0)
                        res = call_func(trial)
                        if res[0] is None:
                            continue
                        ft, x_trial = res
                        if ft < best_local_f - 1e-12:
                            x = x_trial.copy()
                            best_local_f = ft
                            best_local_x = x.copy()
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        for _ in range(3):
                            if n_calls >= budget or check_time():
                                break
                            trial = x + rng.normal(0, 0.1, dim)
                            trial = np.clip(trial, -5.0, 5.0)
                            res = call_func(trial)
                            if res[0] is None:
                                continue
                            ft, x_trial = res
                            if ft < best_local_f - 1e-12:
                                x = x_trial.copy()
                                best_local_f = ft
                                best_local_x = x.copy()
                                improved = True
                                break
                        if not improved:
                            break
                    else:
                        if rng.random() < 0.1:
                            trial = x + rng.normal(0, step * 2, dim)
                            trial = np.clip(trial, -5.0, 5.0)
                            res = call_func(trial)
                            if res[0] is None:
                                continue
                            ft, x_trial = res
                            if ft < best_local_f - 1e-12:
                                x = x_trial.copy()
                                best_local_f = ft
                                best_local_x = x.copy()
            return best_local_x, best_local_f

        rng = np.random.default_rng(12345)

        # Phase 1: Initial stratified sampling
        n_initial = min(15, max(8, budget // 33))
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = np.array([halton(i + 1, b) for b in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
            x = x * 10.0 - 5.0
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])

        for i in range(5):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])

        if best_x is None:
            x = np.zeros(dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])
            if best_x is None:
                return None

        # Phase 2: Bayesian Optimization for global exploration
        remaining = budget - n_calls
        bo_budget = min(int(remaining * 0.3), 120)
        if remaining >= 30 and not check_time():
            bo_x, bo_f = bayesian_optimize(best_x, best_f, bo_budget, rng)
            if bo_f < best_f:
                best_f = bo_f
                best_x = bo_x.copy()

        # Phase 3: CMA-ES for local exploitation
        remaining = budget - n_calls
        if remaining < 20:
            px, pf = local_polish(best_x, remaining, rng)
            update(px, pf)
            return best_x, best_f

        n_restarts_max = 3
        restarts_done = 0
        cma_budget_min = 30

        while remaining >= cma_budget_min and not check_time() and restarts_done < n_restarts_max:
            if restarts_done == 0:
                max_calls = min(remaining - 2 * cma_budget_min, budget - n_calls)
                max_calls = max(max_calls, cma_budget_min)
            else:
                max_calls = min(remaining, budget - n_calls)
                max_calls = max(max_calls, cma_budget_min)

            if max_calls < cma_budget_min:
                break

            if restarts_done == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                x0 = best_x + rng.normal(0, 0.3, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                sigma0 = 0.5

            res_x, res_f, used = cma_es(x0, sigma0, max_calls, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
            remaining = budget - n_calls
            restarts_done += 1

            if restarts_done > 0 and res_f >= best_f - 1e-8:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
                if remaining >= cma_budget_min and not check_time():
                    res_x, res_f, used = cma_es(x0, sigma0, min(remaining, budget - n_calls), rng)
                    if res_f < best_f:
                        best_f = res_f
                        best_x = res_x.copy()
                    remaining = budget - n_calls

        # Phase 4: Local polish
        remaining = budget - n_calls
        if remaining > 5 and not check_time():
            px, pf = local_polish(best_x, min(remaining, 50), rng)
            update(px, pf)

        return best_x, best_f
# EVOLVE-BLOCK-END
