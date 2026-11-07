import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        # Multi-start: evaluate several initial points with different scales
        n_starts = 4
        starts = []
        scales = [0.5, 1.0, 2.0, 4.0]
        for i in range(n_starts):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng.uniform(-4, 4, n) * scales[i]
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            calls += 1
            starts.append((x0, fx))

        if not starts:
            return np.zeros(n), 0.0

        # Pick the best start
        best_idx = min(range(len(starts)), key=lambda i: starts[i][1])
        x = starts[best_idx][0].copy()
        fbest = starts[best_idx][1]

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))  # 8 for n=5
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        sigma = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        inv_C_sqrt = np.eye(n)

        stagnation_count = 0
        restart_threshold = 30
        best_f_so_far = fbest

        remaining = budget - calls
        gen = 0
        while calls < budget and time_ok():
            # Generate lambda candidates
            offspring = []
            for i in range(lam):
                if calls >= budget or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                offspring.append((fi, xi, z))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < fbest:
                fbest = offspring[0][0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Check for restart
            if stagnation_count > restart_threshold and sigma < 0.1:
                # Reset CMA-ES state but keep x
                sigma = 0.5
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)
                stagnation_count = 0
                continue

            # Update mean
            x_old = x.copy()
            x = np.zeros(n)
            for i in range(mu):
                x += weights[i] * offspring[i][1]

            # Compute y_mean
            if sigma > 1e-12:
                y_mean = (x - x_old) / sigma
            else:
                y_mean = np.zeros(n)

            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_mean)
            ps_norm = np.linalg.norm(ps)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * y_mean
            delta = (1 + 2 * max(0, mu_eff / n - 1)) * c_c * (1 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)

            # Ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-4, 10.0)
            gen += 1

        # Final polish with Nelder-Mead if time permits
        time_left = deadline - time.time()
        if time_left > 5.0 and calls < budget:
            try:
                def wrapped_func(x):
                    nonlocal calls
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    return fx

                result = scipy.optimize.minimize(
                    wrapped_func, x, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x = result.x
            except Exception:
                pass

        return x, fbest
