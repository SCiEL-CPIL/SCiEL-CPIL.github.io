import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        d = self.dim
        N = self.budget

        call_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= N or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                return np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to locate the valley
        rng = np.random.default_rng(42)
        n_initial = min(30, N // 10)
        for _ in range(n_initial):
            if time.time() > deadline or call_count >= N:
                break
            x = rng.uniform(-5, 5, d)
            wrapped_func(x)

        if call_count == 0:
            wrapped_func(np.zeros(d))

        # Phase 2: CMA-ES optimization
        # CMA-ES adapts to the anisotropic curvature of Bent Cigar
        # by learning the covariance structure of the search distribution

        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 0.5  # Initial step size
        C = np.eye(d)  # Covariance matrix
        C_inv = np.eye(d)
        ps = np.zeros(d)  # Evolution path for sigma
        pc = np.zeros(d)  # Evolution path for C
        cs = (4 + math.log(d)) / (d + 4)  # Learning rate for sigma
        c1 = (2 + 2 / (d + 1.3)) / ((d + 2) ** 2 + 2)  # Learning rate for C
        cmu = min(1 - c1, (2 * (mu - 1 - c1)) / ((d + 2) ** 2 + 2))  # Learning rate for C from population
        cc = (4 + d / cs) / (d + 4 + 2 / cs)  # Learning rate for pc
        chi_n = math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))  # Expected norm of N(0,I)

        # Population size
        mu = max(1, d // 2 + 1)  # Number of parents
        lambda_pop = 2 * mu  # Population size

        # Precompute weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = math.log((mu + 1) / 2) - math.log(i + 1)
        w = w / w.sum()
        w_max = 2 - w.sum()
        w_max = max(w_max, 1e-10)

        # Main CMA-ES loop
        while call_count < N and time.time() < deadline:
            # Generate population
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
            except np.linalg.LinAlgError:
                L = np.eye(d)

            # Generate lambda_pop candidates
            candidates = []
            candidate_fvalues = []

            for j in range(lambda_pop):
                if call_count >= N or time.time() > deadline:
                    break
                # Standard normal sample
                z = rng.standard_normal(d)
                # Transform to search distribution
                x_j = x_mean + sigma * L @ z
                x_j = np.clip(x_j, -5.0, 5.0)
                f_j = wrapped_func(x_j)
                if call_count < N:  # Only add if we actually made a call
                    candidates.append(x_j)
                    candidate_fvalues.append(f_j)

            if len(candidates) == 0:
                break

            # Sort by fitness
            indices = np.argsort(candidate_fvalues)
            sorted_candidates = [candidates[i] for i in indices]
            sorted_fvalues = [candidate_fvalues[i] for i in indices]

            # Update mean using top mu
            x_new_mean = np.zeros(d)
            for i in range(mu):
                if i >= len(sorted_candidates):
                    break
                x_new_mean += w[i] * sorted_candidates[i]

            # Update evolution paths
            x_diff = x_new_mean - x_mean
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * (C_inv @ x_diff) / sigma
            pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * x_diff / sigma

            # Update sigma
            if np.linalg.norm(ps) / chi_n < 1.4 + 2 / (d + 1):
                sigma *= math.exp(cs / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                # d_sigma = 1 + 2/max(0, d-2)  # Standard value
                # Simplify: use cs/5 as approximation for d_sigma
                sigma *= math.exp(cs / 5.0 * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            delta_hs = 1 if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (d + 1)) * 1.1 else 0
            rank_one_update = np.outer(pc, pc)
            rank_mu_update = np.zeros((d, d))
            for i in range(mu):
                if i >= len(sorted_candidates):
                    break
                y_i = (sorted_candidates[i] - x_mean) / sigma
                rank_mu_update += w[i] * np.outer(y_i, y_i)

            C = (1 + c1 * delta_hs * cc * (2 - cc) - c1 - cmu) * C + \
                c1 * (rank_one_update + (1 - delta_hs) * cc * (2 - cc) * C - np.eye(d)) + \
                cmu * rank_mu_update

            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            C = C + 1e-10 * np.eye(d)

            # Update inverse (expensive, but necessary for accuracy)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(d)

            # Update mean
            x_mean = x_new_mean

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(sigma * L) < 1e-8):
                break

        return best_x, best_f
