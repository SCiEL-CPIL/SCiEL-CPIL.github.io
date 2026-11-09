"""Optimized search for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Sobol global search phase and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, 30s safety margin
        dim = self.dim
        budget = self.budget

        # Sobol quasi-random global search phase
        n_global = min(50, budget // 10)
        x_list = []
        f_list = []

        # Generate Sobol points
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_global)  # in [0,1]^dim
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        for i in range(n_global):
            x = sobol_points[i]
            fx = func(x)
            x_list.append(x)
            f_list.append(fx)
            if time.time() > deadline:
                break

        # Find best point from global search
        best_idx = int(np.argmin(f_list))
        x_best = x_list[best_idx].copy()
        f_best = f_list[best_idx]

        remaining = budget - len(f_list)
        if remaining <= 0:
            return x_best, f_best

        # CMA-ES local exploitation phase
        # Initialize CMA-ES parameters
        sigma = 1.0  # initial step size
        mean = x_best.copy()
        C = np.eye(dim)  # covariance matrix
        Cinv = np.eye(dim)
        Csqrt_inv = np.eye(dim)
        # Compute inverse square root of C
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        Csqr_inv = Cinv_sqrt

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Strategy parameters
        mu = max(1, dim // 2)  # number of selected individuals
        lambda_pop = max(2 * dim, 2 * mu + 1)  # population size
        w = np.zeros(mu)
        w_sum = 0.0
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w_sum = np.sum(w)
        w = w / w_sum
        mueff = 1.0 / np.sum(w ** 2)

        ccs = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cc = (2 + mueff / (dim + 2)) / (dim + 2 + mueff)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Damping for path updates
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES main loop
        call_count = len(f_list)
        for gen in range(remaining):
            if time.time() > deadline:
                break

            # Generate lambda_pop offspring
            ys = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)

            for i in range(lambda_pop):
                # Generate random direction
                z = np.random.randn(dim)
                # Transform using C inverse square root
                y = mean + sigma * (Cinv_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                call_count += 1

                if time.time() > deadline:
                    break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += w[i] * ys_sorted[i]
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                x_best = ys_sorted[0].copy()

            # Update evolution paths
            z_mean = np.zeros(dim)
            for i in range(mu):
                z_mean += w[i] * (Cinv_sqrt @ (ys_sorted[i] - mean)) / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * z_mean

            # Update covariance matrix
            delta_hsig = (1.0 if hsig else 0.0) - (1 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (ys_sorted[i] - mean) / sigma
                rank_mu += w[i] * np.outer(diff, diff)

            C = (1 + c1 * delta_hsig + c1 * cc * (1 - cc) * np.dot(pc, pc) - cs * (2 - cs) * mueff) * C + \
                c1 * rank_one + cmu * rank_mu

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update Cinv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Adaptive sigma: shrink if no improvement
            if gen > 0 and gen % 10 == 0:
                recent_best = np.min(fs_sorted[:min(mu, len(fs_sorted))])
                if recent_best >= f_best:
                    sigma *= 0.95

            # Check convergence
            if sigma < 1e-10 or np.all(np.abs(Cinv_sqrt) < 1e-8):
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
