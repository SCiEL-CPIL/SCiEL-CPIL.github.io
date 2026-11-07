"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive sigma for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        mean = np.zeros(dim)
        sigma = 2.0

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 2:
            lambda_ = 2

        mu = lambda_ // 2

        # Recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / w.sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Threshold for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial function evaluation
        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls = 1

        # Precompute C_inv_sqrt
        C_inv_sqrt = np.eye(dim)

        # Generation counter for h_sigma
        gen = 0

        # Main CMA-ES loop
        while calls + lambda_ <= budget:
            # Generate population
            Z = np.random.standard_normal((lambda_, dim))
            Y = (B * D) @ Z.T
            X = mean + sigma * Y.T
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls += lambda_

            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = X[min_idx].copy()

            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update evolution paths
            diff = (mean - mean_old) / sigma

            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)

            # h_sigma calculation with generation counter
            gen += 1
            # The threshold for h_sigma depends on generation
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*t)) < (1+2/(n+1))*chi_n
            # For simplicity, we use a simplified version
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)

            # Update covariance matrix
            Y_selected = (X_sorted[:mu] - mean_old) / sigma

            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            C += c_mu * (Y_selected.T * w) @ Y_selected

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigendecomposition
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                C_inv_sqrt = (B / D) @ B.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv_sqrt = np.eye(dim)
                sigma *= 0.5

            # Check for convergence
            if sigma < 1e-8:
                break

        # Use remaining budget for local search
        remaining = budget - calls
        if remaining > 0:
            # Nelder-Mead local search
            # Create initial simplex
            simplex = [best_x.copy()]
            step = max(sigma * 0.5, 0.1)
            for i in range(dim):
                x = best_x.copy()
                x[i] += step
                simplex.append(x)
            simplex = np.array(simplex)
            f_simplex = np.array([func(x) for x in simplex])
            calls += dim

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iter = remaining // (dim + 1)
            max_iter = min(max_iter, remaining)

            for _ in range(max_iter):
                if calls >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

                # Update best
                if f_simplex[0] < best_fx:
                    best_fx = f_simplex[0]
                    best_x = simplex[0].copy()

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1

                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1

                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
