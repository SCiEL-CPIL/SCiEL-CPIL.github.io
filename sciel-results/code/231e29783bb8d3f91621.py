import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        a = 2.5
        f0 = evaluate(np.zeros(dim))

        x_est = np.zeros(dim)
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = a
            f_i = evaluate(e_i)
            x_est[i] = (a * a + f0 - f_i) / (2 * a)

        x_est = np.clip(x_est, -5, 5)
        f_est = evaluate(x_est)

        x = best_x.copy()
        fx = best_f

        # Use CMA-ES with small initial sigma since we have a good estimate
        # CMA-ES with lambda=10, mu=5
        rng = np.random.default_rng(12345)
        sigma = 0.1
        lambda_pop = 10
        mu = 5
        weight_sum = 0.0
        for k in range(mu):
            weight_sum += 1.0 / (k + 1)
        weights = np.array([1.0 / (k + 1) for k in range(mu)])
        weights /= weight_sum
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (dim + mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, c_1)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        step_size = np.sqrt(2 * (dim + 1) * (1 - 1 / np.sqrt(2 * (dim + 1))))

        max_iter = 100
        for it in range(max_iter):
            if calls_made + lambda_pop > budget or time.time() > deadline:
                break

            # Generate population
            Z = rng.standard_normal((lambda_pop, dim))
            Y = np.array([x + sigma * (Z[i] @ np.linalg.cholesky(C)) for i in range(lambda_pop)])
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([evaluate(Y[i]) for i in range(lambda_pop)])

            # Sort by fitness
            order = np.argsort(F)
            Z_sorted = Z[order]
            Y_sorted = Y[order]

            # Update mean
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update sigma
            z_sigma = (x_new - x) / (sigma * step_size * np.sqrt(np.trace(C)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1 + 2 / (dim + 1)) * step_size else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / (sigma * step_size)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(p_c, p_c)
            C = C + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x, (Y_sorted[:mu] - x) / (sigma * step_size))[:, None, :] / (sigma ** 2 * step_size ** 2), axis=0)
            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update sigma
            sigma *= np.exp(c_sigma / (2 * dim) * (np.linalg.norm(np.linalg.cholesky(C)) ** 2 - dim))
            sigma = min(sigma, 1.0)

            x = x_new
            fx = F[order[0]]

            if sigma < 1e-14:
                break

        # Final polish with (1+1)-ES
        sigma = 0.01
        rng2 = np.random.default_rng(999)
        for _ in range(200):
            if calls_made >= budget or time.time() > deadline:
                break
            z = rng2.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.9

        return best_x, best_f
