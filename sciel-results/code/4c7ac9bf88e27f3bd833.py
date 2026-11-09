"""CMA-ES with DE-based restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with DE-based restarts for multimodal Weierstrass optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

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
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped)), x_clipped
            except Exception:
                return None, None

        def update(x, f):
            nonlocal best_x, best_f
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES with call tracking. Returns (best_x, best_f, calls_used)."""
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0

            x = x0.copy()
            calls_used = 0
            local_best_x = x.copy()
            local_best_f = np.inf

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if check_time() or calls_used >= max_calls:
                    break
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                res = call_func(x_trial)
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                f, x_eval = res
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()

            # Main loop
            gen = 0
            while calls_used < max_calls and not check_time():
                gen += 1
                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * pop[i]
                x_mean_old = x
                x = x_new

                # Update ps and pc
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x - x_mean_old)
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x - x_mean_old)

                # Update covariance
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Generate new population
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    res = call_func(x_trial)
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f, x_eval = res
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()

                # Convergence check
                if sigma < 1e-10:
                    break

            return local_best_x, local_best_f, calls_used

        def differential_evolution(x0, max_calls, rng):
            """Run jDE with call tracking. Returns (best_x, best_f, calls_used)."""
            n = dim
            lam = min(20, max(8, n * 4))

            F_base = 0.5
            CR_base = 0.7
            sigma_F = 0.1
            sigma_CR = 0.1

            calls_used = 0
            local_best_x = x0.copy()
            local_best_f = np.inf

            # Initialize population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)

            pop[0] = x0
            res = call_func(pop[0])
            if res[0] is None:
                return local_best_x, local_best_f, calls_used
            fvals[0] = res[0]
            calls_used += 1
            if fvals[0] < local_best_f:
                local_best_f = fvals[0]
                local_best_x = pop[0].copy()

            for i in range(1, lam):
                if check_time() or calls_used >= max_calls:
                    break
                pop[i] = rng.uniform(-5, 5, n)
                res = call_func(pop[i])
                if res[0] is None:
                    return local_best_x, local_best_f, calls_used
                fvals[i] = res[0]
                calls_used += 1
                if fvals[i] < local_best_f:
                    local_best_f = fvals[i]
                    local_best_x = pop[i].copy()

            generation = 0
            while calls_used < max_calls and not check_time():
                generation += 1
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                progress = min(1.0, generation / 50.0)
                F_current = F_base + (0.3 - F_base) * progress
                CR_current = CR_base + (0.9 - CR_base) * progress

                new_pop = pop.copy()
                new_fvals = fvals.copy()

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break

                    F_sample = max(0.1, min(1.0, F_current + rng.normal(0, sigma_F)))
                    CR_sample = max(0.1, min(1.0, CR_current + rng.normal(0, sigma_CR)))

                    r1, r2, r3 = rng.choice(lam, size=3, replace=False)
                    while r1 == i:
                        r1 = rng.choice(lam)
                    while r2 == i or r2 == r1:
                        r2 = rng.choice(lam)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = rng.choice(lam)

                    v = pop[r1] + F_sample * (pop[r2] - pop[r3])

                    jrand = rng.integers(n)
                    trial = np.zeros(n)
                    for j in range(n):
                        if rng.random() <= CR_sample or j == jrand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]

                    trial = np.clip(trial, -5, 5)

                    res = call_func(trial)
                    if res[0] is None:
                        return local_best_x, local_best_f, calls_used
                    f_trial = res[0]
                    calls_used += 1

                    if f_trial <= fvals[i]:
                        new_pop[i] = trial
                        new_fvals[i] = f_trial
                        if f_trial < local_best_f:
                            local_best_f = f_trial
                            local_best_x = trial.copy()

                pop = new_pop
                fvals = new_fvals

                if np.std(fvals) < 1e-10:
                    break

                best_in_new = np.argmin(fvals)
                if fvals[best_in_new] > local_best_f:
                    pop[0] = local_best_x
                    fvals[0] = local_best_f

            return local_best_x, local_best_f, calls_used

        def local_polish(x_start, max_calls, rng):
            """(1+1)-ES local refinement."""
            n = dim
            x = x_start.copy()
            res = call_func(x)
            if res[0] is None:
                return x, np.inf, 0
            f = res[0]
            calls_used = 1
            sigma = 0.1

            while calls_used < max_calls and not check_time():
                z = rng.standard_normal(n)
                trial = x + sigma * z
                trial = np.clip(trial, -5, 5)

                res = call_func(trial)
                if res[0] is None:
                    break
                f_trial = res[0]
                calls_used += 1

                if f_trial < f:
                    x = trial
                    f = f_trial
                    sigma *= 1.5
                else:
                    sigma *= 0.7

                if sigma < 1e-8:
                    break

            return x, f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling
        n_initial = min(10, budget // 50)
        for i in range(n_initial):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            res = call_func(x)
            if res[0] is None:
                break
            f, x_eval = res
            update(x_eval, f)

        if best_x is None:
            x = np.zeros(dim)
            res = call_func(x)
            if res[0] is not None:
                update(res[1], res[0])
            if best_x is None:
                return None

        remaining = budget - n_calls
        if remaining < 10:
            return best_x, best_f

        # Phase 2: CMA-ES from best point (main global search)
        # Allocate 50% to CMA-ES, 30% to DE restart, 20% to local polish
        cma_budget = int(remaining * 0.5)
        de_budget = int(remaining * 0.3)
        polish_budget = remaining - cma_budget - de_budget

        # Run CMA-ES
        res_x, res_f, used = cma_es(best_x.copy(), 2.0, cma_budget, rng)
        if res_f < best_f:
            best_f = res_f
            best_x = res_x.copy()

        # Phase 3: DE-based restart if budget remains
        if de_budget > 10 and not check_time():
            # Use DE from a perturbed version of best point
            x_start = best_x + rng.normal(0, 1.0, dim)
            x_start = np.clip(x_start, -5, 5)
            res_x, res_f, used = differential_evolution(x_start, de_budget, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        # Phase 4: Local polish
        if polish_budget > 5 and not check_time():
            res_x, res_f, used = local_polish(best_x.copy(), polish_budget, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
