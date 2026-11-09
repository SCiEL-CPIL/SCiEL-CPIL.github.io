import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s/330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_left(n=5):
            return eval_count < budget - n

        # ---- Phase 1: Diverse early sampling for fast initial AOCC progress ----
        samples = []
        for _ in range(12):
            samples.append(rng.uniform(-4.5, 4.5, d))
        for _ in range(8):
            samples.append(rng.uniform(-2.5, 2.5, d))
        for _ in range(6):
            samples.append(rng.uniform(-1.0, 1.0, d))
        samples.append(np.zeros(d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.normal(0, 1.5, d))
        samples.append(rng.normal(0, 0.8, d))

        n_random = min(len(samples), budget // 6)
        for x in samples[:n_random]:
            if not budget_left(80) or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Nelder–Mead simplex (robust to hidden rotation) ----
        def nelder_mead(x0, max_iter=250, init_step=0.5):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or not budget_left(30):
                return x0

            simplex = [x0.copy()]
            for i in range(d):
                v = x0.copy()
                v[i] += init_step
                simplex.append(v)
            f_values = [evaluate(v) for v in simplex]

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if not time_ok() or not budget_left(30):
                    break

                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                if f_values[-1] - f_values[0] < 1e-11 * (1 + abs(f_values[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_values[-1]

                x_r = np.clip(centroid + alpha * (centroid - worst), -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < f_values[0]:
                    x_e = np.clip(centroid + gamma * (centroid - worst), -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1], f_values[-1] = x_e, f_e
                    else:
                        simplex[-1], f_values[-1] = x_r, f_r
                elif f_r < f_values[-2]:
                    simplex[-1], f_values[-1] = x_r, f_r
                else:
                    if f_r < f_worst:
                        x_c = np.clip(centroid + rho * (x_r - centroid), -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c <= f_r:
                            simplex[-1], f_values[-1] = x_c, f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])
                    else:
                        x_c = np.clip(centroid - rho * (centroid - worst), -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1], f_values[-1] = x_c, f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])

            idx = int(np.argmin(f_values))
            return simplex[idx]

        x_nm = nelder_mead(best_x.copy(), max_iter=250, init_step=0.5)

        # ---- Phase 3: CMA-ES (pop=4) with cumulative step-size control ----
        # Starts from the current best to exploit locality while the adaptive
        # covariance + step-size handles the hidden rotation/shift.
        def cma_es(x0, max_iter=60, sigma0=0.5):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or not budget_left(25):
                return x0

            n = d
            lam = 4
            mu = max(1, lam // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = int(1.0 / np.sum(w**2))
            c_sigma = (mueff + 2) / ((n + 2)**2 + mueff)
            d_sigma = 1.0 + 2 * max(0.0, (mueff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - c_c + mueff / n) / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

            mean = np.asarray(x0, dtype=np.float64).copy()
            sigma = float(sigma0)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            for _ in range(max_iter):
                if not time_ok() or not budget_left(25):
                    break

                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

                # Sample population
                pop = []
                for k in range(lam):
                    z = rng.standard_normal(n)
                    pop.append(mean + sigma * (L @ z))

                pop = np.array(pop)
                fvals = np.array([evaluate(np.clip(p, -5.0, 5.0)) for p in pop])

                order = np.argsort(fvals)
                y = (pop[order] - mean) / sigma
                y_best = y[:mu]

                # Update covariance (rank-mu)
                v = y_best.T @ w
                delta_sig = (1 - c_sigma) * (np.linalg.norm(ps) / chi_n - 1.0)
                C = ((1 - (c_1 + c_mu) * delta_sig) * C
                     + c_1 * (np.outer(pc, pc) - C)
                     + c_mu * (v @ v.T))
                # Symmetrize + ensure positive-definite-ish
                C = 0.5 * (C + C.T)
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-9 * np.eye(n)

                # Update step size (cumulative control)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (v @ np.sqrt(np.diag(C))) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = float(np.clip(sigma, 1e-12, 50.0))

                # Update mean (weighted recombination)
                mean = mean + sigma * (w @ y_best.T)
                mean = np.clip(mean, -5.0, 5.0)

                # Update evolution path pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (v @ np.sqrt(np.diag(C))) / sigma

            return mean

        if budget_left(40) and time_ok():
            x_cma = cma_es(best_x.copy(), max_iter=60, sigma0=0.4)

        # ---- Phase 4: Second Nelder–Mead polish ----
        if budget_left(30) and time_ok():
            x_nm2 = nelder_mead(best_x.copy(), max_iter=150, init_step=0.1)

        # ---- Phase 5: Coordinate-wise adaptive refinement (f14 separability) ----
        refine_x = best_x.copy()
        refine_f = best_f
        for i in range(d):
            if not budget_left(10) or not time_ok():
                break
            for step_size in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
                if not budget_left(10) or not time_ok():
                    break
                improved = False
                for direction in (1.0, -1.0):
                    if not budget_left(10) or not time_ok():
                        break
                    test_x = refine_x.copy()
                    test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()
                        improved = True
                if not improved:
                    break

        # ---- Phase 6: Final fine Nelder–Mead ----
        if budget_left(15) and time_ok():
            x_final = nelder_mead(refine_x.copy(), max_iter=100, init_step=0.03)

        return best_x, best_f
