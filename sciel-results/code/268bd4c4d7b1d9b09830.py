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

        rng = np.random.default_rng(123)

        # Global best
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls = 1

        def time_ok():
            return time.time() < deadline

        def nelder_mead(x0, f0, max_evals, step=0.5):
            nonlocal calls, x_best, f_best
            if calls >= budget or not time_ok():
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            simplex = np.zeros((dim + 1, dim))
            fvals = np.zeros(dim + 1)
            simplex[0] = x0
            fvals[0] = f0

            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex[i + 1] = xi
                fvals[i + 1] = func(xi)
                calls += 1
                if calls >= budget or not time_ok():
                    idx = np.argmin(fvals)
                    if fvals[idx] < f_best:
                        f_best = fvals[idx]
                        x_best = simplex[idx].copy()
                    return

            for _ in range(max_evals):
                if calls >= budget or not time_ok():
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1

                if f_ref < fvals[0]:
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        simplex[-1], fvals[-1] = x_exp, f_exp
                    else:
                        simplex[-1], fvals[-1] = x_ref, f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1], fvals[-1] = x_ref, f_ref
                else:
                    if f_ref < fvals[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1], fvals[-1] = x_con, f_con
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or not time_ok():
                                break

                idx = np.argmin(fvals)
                if fvals[idx] < f_best:
                    f_best = fvals[idx]
                    x_best = simplex[idx].copy()

        def cma_es(x0, sigma0, lam, max_evals):
            nonlocal calls, x_best, f_best
            if calls >= budget or not time_ok():
                return

            n = dim
            C = np.eye(n)
            sigma = sigma0
            x_mean = x0.copy()
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            c_sigma = (5 + lam/n) / (n + 5 + 2*lam/n)
            d_sigma = 1 + 2*max(0, np.sqrt((lam-1)/(n+1)) - 1) + c_sigma
            c_c = (4 + lam/n) / (n + 4 + 3*lam/n)
            c_1 = 2 / ((n + 1.3)**2 + lam)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((n + 2)**2 + mu_eff))
            mu_eff = lam / 2.0
            w = np.zeros(lam)
            w[:int(mu_eff)] = np.log(lam/2) - np.log(np.arange(1, int(mu_eff)+1))
            w /= w.sum()
            mu_w = w @ np.arange(lam)
            c_1 += 2 * (mu_w - 1) * (mu_w - 1) / ((n+2)**2) - c_1

            for gen in range(max_evals // lam):
                if calls >= budget or not time_ok():
                    break

                B, D = np.linalg.eigh(C)
                B = B[:, ::-1]
                D = np.sqrt(D[::-1])
                M = B / D

                ys = np.zeros((lam, n))
                xs = np.zeros((lam, n))
                fs = np.zeros(lam)

                for i in range(lam):
                    y = rng.standard_normal(n)
                    ys[i] = y
                    x = x_mean + sigma * (M @ y)
                    x = np.clip(x, -5, 5)
                    xs[i] = x
                    fs[i] = func(x)
                    calls += 1
                    if calls >= budget or not time_ok():
                        break

                if calls >= budget or not time_ok():
                    idx = np.argmin(fs)
                    if fs[idx] < f_best:
                        f_best = fs[idx]
                        x_best = xs[idx].copy()
                    return

                order = np.argsort(fs)
                xs = xs[order]
                ys = ys[order]
                fs = fs[order]

                x_new = (w[:, None] * xs[:int(mu_eff)]).sum(axis=0)
                y_new = (w[:, None] * ys[:int(mu_eff)]).sum(axis=0)

                h_sigma = 0
                if np.linalg.norm(y_new) < (1 + 1/(2*c_sigma)) * chi_n:
                    h_sigma = 1
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma)/chi_n - 1))

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_new
                delta_h = (1 - (1-c_c)**(2*gen+1)) / (1 - (1-c_c)**2)
                C = (1 - c_1 - c_mu + h_sigma * c_c * (2 - c_c) * delta_h) * C \
                    + c_1 * (np.outer(p_c, p_c) + (h_sigma * c_c * (2 - c_c) * delta_h - 1) * C) \
                    + c_mu * (w[:, None] * (xs[:int(mu_eff)] - x_mean) @ ((xs[:int(mu_eff)] - x_mean).T * w[None, :int(mu_eff)]))

                x_mean = x_new
                idx = np.argmin(fs)
                if fs[idx] < f_best:
                    f_best = fs[idx]
                    x_best = xs[idx].copy()

        # Phase 1: CMA-ES for exploration
        if calls < budget and time_ok():
            x0 = x_best.copy()
            cma_es(x0, 2.0, 12, budget - calls - dim - 1)

        # Phase 2: Nelder-Mead refinement
        if calls < budget and time_ok():
            nelder_mead(x_best, f_best, budget - calls)

        # Phase 3: If budget remains, do a few more random starts with NM
        remaining = budget - calls
        if remaining > 10 and time_ok():
            n_starts = min(3, remaining // 10)
            for _ in range(n_starts):
                if calls >= budget or not time_ok():
                    break
                x_rand = rng.uniform(-5, 5, dim)
                f_rand = func(x_rand)
                calls += 1
                if f_rand < f_best:
                    f_best = f_rand
                    x_best = x_rand.copy()
                nelder_mead(x_best, f_best, budget - calls)

        return x_best, f_best
