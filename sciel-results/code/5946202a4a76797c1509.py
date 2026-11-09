"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with diverse initialization for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(y):
            nonlocal best_x, best_f, calls
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            return fy

        # Phase 1: Diverse initial sampling for early AOCC progress
        n_init = min(20, self.budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                y = np.zeros(dim)
            else:
                y = rng.uniform(-5, 5, dim)
            eval_point(y)

        if calls >= self.budget or best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        # CMA-ES parameters for dim=5
        lambda_ = max(4, 4 + int(3 * np.log(dim)))  # ~13 for dim=5
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize CMA-ES state from best point found
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-20))
        B = eigvecs

        # Track stagnation for potential restart
        best_f_window_start = best_f
        calls_window_start = calls
        window_size = 3 * lambda_

        while calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            lam_cur = min(lambda_, remaining)
            if lam_cur < 2:
                break

            # Generate and evaluate population
            offspring = []
            fitnesses = []
            for i in range(lam_cur):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = eval_point(y)
                offspring.append(y)
                fitnesses.append(fy)

            if len(offspring) < mu:
                break

            # Sort by fitness
            sorted_idx = np.argsort(fitnesses)
            mu_actual = min(mu, len(offspring))
            selected = [offspring[i] for i in sorted_idx[:mu_actual]]
            w_actual = w[:mu_actual]
            w_actual = w_actual / w_actual.sum()

            # Update mean
            mean_old = mean.copy()
            mean = sum(w_actual[i] * selected[i] for i in range(mu_actual))

            # Update step size (sigma)
            y_sigma = (B @ (D @ pc)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            y = np.array([(sel - mean_old) / sigma for sel in selected])
            delta_hsig = (1 - (np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * chi_n)) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = sum(w_actual[i] * np.outer(y[i], y[i]) for i in range(mu_actual))
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update pc
            hsig = np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma

            # Check for stagnation and restart if needed
            if calls - calls_window_start >= window_size:
                if best_f - best_f_window_start < 1e-6:
                    # Stagnation detected, try a restart with larger sigma
                    sigma = 2.0
                    C = np.eye(dim)
                    D = np.ones(dim)
                    B = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    best_f_window_start = best_f
                    calls_window_start = calls
                else:
                    best_f_window_start = best_f
                    calls_window_start = calls

        return best_x, best_f
# EVOLVE-BLOCK-END
