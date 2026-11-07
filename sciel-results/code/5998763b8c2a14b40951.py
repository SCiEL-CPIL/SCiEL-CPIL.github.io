import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer for BBOB f21 (Gallagher 101 peaks).
    Uses a population-based evolutionary strategy with adaptive covariance
    for better exploration of the multi-modal landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # Clip to domain
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        dim = self.dim
        pop_size = max(4, dim + 3)  # Standard CMA-ES population size
        sigma = 1.0
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        c_sigma = 1.0 / (pop_size + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size // 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial phase: LHS sampling for diversity
        n_initial = min(20, self.budget // 15)

        def generate_lhs(n, dim, rng):
            points = np.zeros((n, dim))
            for j in range(dim):
                perm = rng.permutation(n)
                points[:, j] = perm / n + rng.random(n) / n
            points = points * 10.0 - 5.0
            return points

        initial_points = generate_lhs(n_initial, dim, rng)
        initial_vals = []
        for p in initial_points:
            if calls_made >= self.budget or time_left() < 10:
                break
            v = evaluate(p)
            initial_vals.append((v, p.copy()))

        # Sort and pick best as starting point for CMA-ES
        initial_vals.sort(key=lambda x: x[0])
        if initial_vals:
            x_mean = initial_vals[0][1].copy()
            best_f = initial_vals[0][0]
            best_x = initial_vals[0][1].copy()

        # CMA-ES main loop
        while calls_made < self.budget - 5 and time_left() > 5:
            # Generate population
            D = np.linalg.cholesky(C)
            B = D.T / np.sqrt(np.diag(D @ D.T))
            
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                offspring.append(x_i)

            # Evaluate offspring
            fitnesses = []
            for x_i in offspring:
                if calls_made >= self.budget or time_left() < 2:
                    break
                f_i = evaluate(x_i)
                fitnesses.append((f_i, x_i))

            if len(fitnesses) < pop_size:
                break

            # Sort by fitness
            fitnesses.sort(key=lambda x: x[0])

            # Update best
            if fitnesses[0][0] < best_f:
                best_f = fitnesses[0][0]
                best_x = fitnesses[0][1].copy()

            # Update mean
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            x_new = sum(weights[i] * fitnesses[i][1] for i in range(mu))

            # Update step size (sigma)
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * np.linalg.inv(B) @ (x_new - x_mean) / sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
            delta_sig = (1 - (1 - c_c) ** 2) * (np.linalg.norm(pc) / chi_n - 1)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sig * C)
            for i in range(mu):
                y_i = (fitnesses[i][1] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            x_mean = x_new

            # Check for convergence
            if sigma < 1e-6:
                break

        # Final polish
        if calls_made < self.budget and time_left() > 2 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.05
            no_improve = 0
            max_no_improve = 20

            while calls_made < self.budget and time_left() > 1 and no_improve < max_no_improve:
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 1.1
                    no_improve = 0
                else:
                    sigma_polish *= 0.9
                    no_improve += 1

        return best_x, best_f
