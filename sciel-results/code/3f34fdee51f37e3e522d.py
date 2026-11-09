# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based continuous optimizer with Halton initialization.
    
    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust
    global and local optimization. CMA-ES adapts the search distribution's
    covariance matrix to the problem's landscape, making it highly effective
    for ill-conditioned, rotated, and multimodal functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Phase 1: Coarse Halton sampling for initial global coverage
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        n_halton = 20
        halton_points = np.zeros((n_halton, dim))
        for i in range(1, n_halton + 1):
            for j in range(dim):
                base = primes[j % len(primes)]
                halton_points[i - 1, j] = halton_sequence(i, base)
        halton_points = halton_points * 10.0 - 5.0

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        for pt in halton_points:
            if calls >= budget or time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        lambda_ = max(4 + int(3 * np.log(dim)), 8)  # Population size
        mu = lambda_ // 2  # Selection size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 0.5
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Evaluate initial population
        def evaluate_population():
            nonlocal calls, best_f, best_x
            pop = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                step = B @ (D * rng.standard_normal(dim))
                pop[i] = np.clip(x_mean + sigma * step, -5, 5)
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            return pop, fvals

        # Main CMA-ES loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate and evaluate population
            pop, fvals = evaluate_population()
            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Weighted mean
            x_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)

            # Update sigma (CSA - Cumulative Step Adaptation)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * B @ (D * (x_new - x_mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / lambda_)) / chi_n) < 1.4 + 2 / (dim + 1)
            ps *= hsig

            # Update pc (CMA)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_mean) / sigma

            # Update covariance matrix
            delta_mu = (1 - (1 - hsig) * c_c) * pc * pc.T / sigma ** 2 - np.eye(dim)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = B @ (D * (pop_sorted[i] - x_mean) / sigma)
                rank_mu += weights[i] * np.outer(z, z)

            C = (1 + c_1 * delta_mu + c_mu * rank_mu) * (B @ np.diag(D ** 2) @ B.T)

            # Eigen-decomposition for step size
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update mean
            x_mean = x_new

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(D) < 1e-10):
                break

            # Stagnation check: if no improvement in recent generations, restart
            if np.min(fvals_sorted) >= best_f + 1e-10 and calls > budget * 0.3:
                # Try a restart from a different location
                if rng.random() < 0.3:
                    x_mean = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    sigma = 1.0
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    fx = func(x_mean)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
