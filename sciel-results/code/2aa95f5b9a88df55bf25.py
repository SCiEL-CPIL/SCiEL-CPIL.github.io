# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """Population-based CMA-ES with an explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.monotonic() + 280.0  # safety margin under the 330 s limit

        rng = np.random.default_rng(0)

        # --- CMA-ES hyperparameters ---
        lambda_ = 4 + int(math.floor(3.0 * math.log(dim)))  # = 10 for dim=5
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff + 2)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 2.0 / (dim + 1)) /
                   ((dim + 2) ** 2 + mu_eff))
        chi_N = math.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # --- initial state ---
        x_m = np.zeros(dim)
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # initial point
        x0 = rng.uniform(-5.0, 5.0, dim)
        best_x = x0.copy()
        best_f = evaluate(x0)

        while calls < budget:
            if time.monotonic() >= deadline:
                break
            if lambda_ > budget - calls:
                break

            # sample population
            ys = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                ys.append(x_m + sigma * (B @ (D * z)))
            ys = np.array(ys)

            # evaluate (stop if we would exceed budget)
            fs = []
            for y in ys:
                if calls >= budget:
                    break
                fs.append(evaluate(y))
            fs = np.array(fs)
            if len(fs) < 2:
                break

            # update best
            i_best = int(np.argmin(fs))
            if fs[i_best] < best_f:
                best_f = fs[i_best]
                best_x = ys[i_best].copy()

            # select top mu
            order = np.argsort(fs)[:mu]
            y_sel = ys[order]
            w_sel = w

            # new mean (weighted recombination)
            x_new = y_sel.T @ w_sel

            # update covariance via evolution path
            y_diff = (x_new - x_m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff) * y_diff
            h_sigma = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** 2)
                       / chi_N < 1.4 + 2.0 / (dim + 1))
            p_c = (1 - c_c) * p_c + (h_sigma * c_c) * y_diff
            c1 = c_1 * (2 - c_1 - c_mu)
            C = ((1 - c1 - c_mu) * C
                 + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim))
                 + c_mu * np.sum(w_sel[:, None, None] * np.outer(y_sel, y_sel), axis=0))
            C = 0.5 * (C + C.T)

            # eigen-decomposition for sampling
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.clip(D2, 1e-20, None))
            B = B2
            D = D2

            # sigma adaptation
            sigma *= math.exp(c_sigma / chi_N *
                              (np.linalg.norm(p_sigma) / chi_N - 1.0))
            sigma = float(np.clip(sigma, 1e-8, 50.0))

            x_m = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
