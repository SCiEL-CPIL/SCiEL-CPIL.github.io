"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard and local polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 2.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cs = (mueff + 1.5) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) / (1 - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        E_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 3.0 / (32 * dim * dim))

        # Track generation count for hsig
        gen = 0

        while calls < budget and time_left():
            # Check if we can afford a full generation
            if calls + lam > budget:
                break

            # Generate population
            L = np.linalg.cholesky(C)
            population = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                population.append((fy, y))

            if calls >= budget:
                break

            gen += 1

            # Sort population by fitness
            population.sort(key=lambda x: x[0])

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]

            # Compute y_old (normalized difference)
            L_norm = np.linalg.norm(L)
            if L_norm > 1e-16:
                y_old = (new_mean - mean) / (sigma * L_norm)
            else:
                y_old = np.zeros(dim)

            # Update step size control vector ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_old

            # hsig condition
            hsig_thresh = (1.4 + 2 / (dim + 1)) * E_norm
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * gen)) < hsig_thresh

            # Update pc
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_old

            # Update covariance matrix
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                if L_norm > 1e-16:
                    z_i = (population[i][1] - mean) / (sigma * L_norm)
                else:
                    z_i = np.zeros(dim)
                rank_mu += weights[i] * np.outer(z_i, z_i)

            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
            C = (C + C.T) / 2.0

            # Update sigma
            sigma = sigma * np.exp(cs * (norm_ps / E_norm - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update mean
            mean = new_mean

            # Check for convergence
            if sigma < 1e-10:
                break

        # Use remaining budget for local polishing with Nelder-Mead
        if calls < budget and time_left():
            remaining = budget - calls
            # Simple local search: perturb best_x with decreasing steps
            step = 0.5
            while calls < budget and time_left() and step > 1e-4:
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if calls >= budget or not time_left():
                            break
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < best_f:
                            best_x = x_try.copy()
                            best_f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
            # Final fine-tuning
            step = 1e-3
            while calls < budget and time_left() and step > 1e-6:
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if calls >= budget or not time_left():
                            break
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < best_f:
                            best_x = x_try.copy()
                            best_f = f_try
                            improved = True
                if not improved:
                    step *= 0.5

        return best_x, best_f

# EVOLVE-BLOCK-END
