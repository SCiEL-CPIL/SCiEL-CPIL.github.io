"""CMA-ES with restarts for BBOB f20 (Schwefel) optimization."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, integrated restart rule, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() < deadline

        # CMA-ES parameters
        lambda_ = 4  # population size
        mu = lambda_ // 2  # number of weights
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        # Start with a reasonable step size for [-5,5]^dim
        sigma = 2.0

        # Mean initialization
        m = rng.uniform(-5, 5, dim)

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute square root of C
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Update counters
        d_sigma = 1 + 0.3 * min((mu_eff - 1) / 2, dim / 2)
        d_cov = 2 + 2 * min(mu_eff / (dim + 2), 1) - 1 / (dim + 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_cov = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov_neg = (2 - c_cov + c_sigma) / (dim + 2)

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Evaluate initial point
        best_x = m.copy()
        best_f = func(m)
        calls_made += 1

        # Generate initial population
        for _ in range(lambda_):
            if not check_time():
                break
            z = rng.standard_normal(dim)
            x = m + sigma * C_sqrt @ z
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        # We need to collect all evaluated points
        # Let's restructure: collect population, evaluate, sort

        # Restart logic
        max_evals_per_run = 200
        total_calls = calls_made
        restart_count = 0
        max_restarts = 5

        while total_calls < budget and restart_count <= max_restarts and check_time():
            # Reset CMA-ES state for this run
            m = best_x.copy()
            sigma = 2.0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Evaluate initial
            if total_calls < budget and check_time():
                fx = func(m)
                total_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = m.copy()

            # Main loop for this run
            run_calls = 0
            no_improve = 0
            prev_best = best_f

            while run_calls < max_evals_per_run and total_calls < budget and check_time():
                # Generate population
                pop = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    x = m + sigma * C_sqrt @ z
                    x = np.clip(x, -5, 5)
                    pop.append(x)

                # Evaluate population
                fvals = np.array([func(x) for x in pop])
                total_calls += lambda_
                run_calls += lambda_

                # Track best
                for i in range(lambda_):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                # Sort by fitness
                idx = np.argsort(fvals)
                y_sorted = np.array([pop[i] for i in idx])
                f_sorted = fvals[idx]

                # Update mean
                m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

                # Update sigma
                y_mean = C_inv_sqrt @ (m_new - m)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 50)

                # Update covariance
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (run_calls + 1) / (dim + c_sigma)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                p_c = (1 - c_cov) * p_c + h_sigma * np.sqrt(c_cov * (2 - c_cov) * mu_eff) * y_mean
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = C_inv_sqrt @ (y_sorted[i] - m)
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                C = (1 + c_cov * c_cov_neg - c_cov) * C + c_cov * np.outer(p_c, p_c) + c_cov_neg * rank_mu

                # Ensure C is positive definite
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if np.min(eigvals) < 1e-10:
                        eigvals = np.maximum(eigvals, 1e-10)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_sqrt = eigvecs @ np.sqrt(np.diag(eigvals))
                    C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

                m = m_new

                # Check for improvement
                if best_f < prev_best - 1e-10:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1

                # Restart if no improvement for a while
                if no_improve > 30:
                    break

            # Check if we should restart
            if total_calls < budget and check_time():
                restart_count += 1
                # For next restart, initialize near best found
                # Keep best_x as the new starting point

        return best_x, best_f
