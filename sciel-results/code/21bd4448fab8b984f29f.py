import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0  # Safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal used, best_f, best_x
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        # CMA-ES Implementation
        class CMAES:
            def __init__(self, x0, sigma0, budget, func, dim, global_best_x, global_best_f, used_counter, deadline_check):
                self.x = x0.copy()
                self.sigma = sigma0
                self.C = np.eye(dim)
                self.p_sigma = np.zeros(dim)
                self.p_c = np.zeros(dim)
                self.chi_N = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                self.mu = int(np.floor(dim / 2))
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.mueff = 1 / np.sum(self.weights ** 2)
                self.c_sigma = (self.mueff + 2) / (dim + self.mueff + 5)
                self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.c_sigma
                self.c_c = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
                self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
                self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
                self.budget = budget
                self.func = func
                self.dim = dim
                self.global_best_x = global_best_x
                self.global_best_f = global_best_f
                self.used_counter = used_counter
                self.deadline_check = deadline_check
                self.rng = np.random.default_rng(int(np.sum(x0 * 12345.6789) + 42))

            def step(self):
                if self.used_counter[0] >= self.budget or self.deadline_check():
                    return False

                # Sampling
                B = self.rng.standard_normal((self.mu, self.dim))
                # Whitening transform
                B = B @ np.linalg.cholesky(self.C).T
                # Rank-mu update of the mean
                y_w = np.average(B, axis=0, weights=self.weights)
                self.x = self.x + self.sigma * y_w

                # Evaluate
                ys = []
                for i in range(self.mu):
                    x_i = self.x + self.sigma * B[i]
                    f_i = self.func(np.clip(x_i, -5, 5))
                    self.used_counter[0] += 1
                    ys.append((f_i, x_i))
                    if f_i < self.global_best_f[0]:
                        self.global_best_f[0] = f_i
                        self.global_best_x[0] = x_i.copy()

                # Sort
                ys.sort(key=lambda t: t[0])

                # Update
                y_w = np.zeros(self.dim)
                for i in range(self.mu):
                    y_w += self.weights[i] * (ys[i][1] - self.x) / self.sigma

                # Update sigma
                self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                               np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * y_w / np.linalg.norm(self.C ** 0.5 @ y_w + 1e-16)
                self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.p_sigma) / self.chi_N - 1))

                # Update covariance matrix
                self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * y_w
                hsig = (np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.used_counter[0] + 1))) <
                        (1.4 + 2 / (self.dim + 1)) * self.chi_N)
                delta_sig = (1 - hsig) * np.sqrt((self.mueff - 1) * (1 + 2 * (self.dim + 1) / (self.mueff - 1)))
                self.C = (1 + self.c_1 + self.c_mu - self.c_1) * self.C + \
                          self.c_1 * (self.p_c @ self.p_c.T - self.C) + \
                          self.c_mu / self.sigma ** 2 * sum(self.weights[i] * np.outer(ys[i][1] - self.x, ys[i][1] - self.x) for i in range(self.mu))

                # Symmetrize C
                self.C = (self.C + self.C.T) / 2

                # Check convergence
                if self.sigma < 1e-10:
                    return False

                return True

        # Multi-start strategy
        # Split budget into restarts
        # 5D problem, CMA-ES typically needs ~10-20 * D * ln(D) ~ 80-160 evals to converge
        # Let's try 3 restarts: 200, 150, 150
        restart_budgets = [200, 150, 150]
        # Adjust to fit total budget
        while sum(restart_budgets) > total_budget:
            restart_budgets[-1] -= 10
            if restart_budgets[-1] <= 0:
                restart_budgets.pop()
        while sum(restart_budgets) < total_budget and len(restart_budgets) < 10:
            restart_budgets.append(50)

        # Initial best is center
        evaluate(np.zeros(dim))

        for i, rb in enumerate(restart_budgets):
            if used >= total_budget or check_deadline():
                break

            # Initialize CMA-ES
            # Start near the current best point, with a reasonable initial sigma
            # If this is the first restart, start at 0. Subsequent restarts can start at best_x or random
            if i == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Perturb the best point slightly to escape local minima
                x0 = best_x + np.random.default_rng(i * 12345).normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            cmaes = CMAES(x0, sigma0, rb, func, dim, [best_x], [best_f], [used], check_deadline)

            # Run CMA-ES
            while cmaes.step():
                if used >= total_budget or check_deadline():
                    break

        return best_x, best_f
