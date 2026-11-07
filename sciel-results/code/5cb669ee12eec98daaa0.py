"""
Hybrid CMA-ES + Local Refinement optimizer for BBOB f3 (Rastrigin).
Uses CMA-ES for global exploration and a lightweight coordinate-descent
refinement for local exploitation, adapting to rotated/shifted landscapes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))

        # State variables
        x_mean = np.zeros(n)
        sigma = 1.5
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def time_check():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def local_refine(x, max_evals=20):
            """
            Lightweight coordinate descent refinement.
            Tries small steps in each dimension and accepts if better.
            Uses a more aggressive step reduction to converge faster.
            """
            nonlocal calls_made, best_f, best_x
            x = x.copy()
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

            step = max(sigma * 0.05, 0.01)
            for _ in range(max_evals):
                if calls_made >= budget or (time.time() - start_time) > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls_made >= budget or (time.time() - start_time) > deadline:
                        break
                    # Try positive step
                    x_try = x.copy()
                    x_try[i] += step
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < f:
                        x = x_try
                        f = f_try
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        # Try negative step
                        if calls_made >= budget or (time.time() - start_time) > deadline:
                            break
                        x_try = x.copy()
                        x_try[i] -= step
                        f_try = func(x_try)
                        calls_made += 1
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                if not improved:
                    # No improvement, shrink step
                    step *= 0.4
                    if step < 1e-7:
                        break

            return x, f

        # Initial evaluation at origin
        x0 = np.zeros(n)
        f0 = func(x0)
        calls_made += 1
        best_f = f0
        best_x = x0.copy()

        # Try a few random initial points to find a better starting point
        for _ in range(5):
            if calls_made >= budget or (time.time() - start_time) > deadline:
                break
            x_rand = np.random.uniform(-3, 3, size=n)
            f_rand = func(x_rand)
            calls_made += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        # CMA-ES main loop
        while not time_check():
            # 1. Sample new points
            ys = []
            for k in range(lam):
                z = np.random.standard_normal(n)
                y = B @ (D * z)
                ys.append(y)

            xs = x_mean + sigma * np.array(ys)

            # 2. Evaluate
            remaining = budget - calls_made
            num_evals = min(lam, remaining)
            if num_evals <= 0:
                break

            fitnesses = []
            eval_xs = []
            for i in range(num_evals):
                x = xs[i]
                f = func(x)
                fitnesses.append(f)
                eval_xs.append(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            if num_evals < lam:
                break

            fitnesses = np.array(fitnesses)
            eval_xs = np.array(eval_xs)

            # 3. Sort by fitness
            order = np.argsort(fitnesses)
            xs_sorted = eval_xs[order]
            fitnesses_sorted = fitnesses[order]

            # 4. Update mean
            new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)

            # 5. Update sigma
            y_mean = (new_mean - x_mean) / sigma
            c_hs = (1.1 + 2.0 / (n + 1.7))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
            norm_y_mean = np.linalg.norm(y_mean)
            sqrt_trace_C = np.sqrt(np.trace(C))
            ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0

            p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made / lam))) < (1.4 + 2 / (n + 1)) * chi_n)

            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # 6. Update covariance matrix C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            rank_one = np.outer(p_c, p_c)

            ys_array = np.array(ys)
            ys_sorted = ys_array[order][:mu]
            y_diffs = ys_sorted - y_mean
            rank_mu = np.sum(weights[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)

            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            C = (C + C.T) / 2

            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)

            x_mean = new_mean

            # Local refinement on best point found so far (less frequent to save budget)
            gen_count = calls_made // lam
            if best_x is not None and sigma > 0.05 and gen_count % 3 == 0:
                _, _ = local_refine(best_x, max_evals=15)

            if sigma < 1e-12:
                break

        # Final high-precision polish pass
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            if remaining >= 10:
                _, _ = local_refine(best_x, max_evals=min(remaining, 40))

        return best_x, best_f

# EVOLVE-BLOCK-END
