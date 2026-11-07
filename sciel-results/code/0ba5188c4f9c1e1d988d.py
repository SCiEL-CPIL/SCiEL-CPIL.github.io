"""Rewritten optimizer: multi-restart CMA-ES with time guard and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and final local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 20s safety margin

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64).clip(-5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Multi-restart CMA-ES ---
        rng = np.random.default_rng(42)

        # Determine number of restarts and budget per restart
        # Use ~60% of budget for exploration, ~40% for final polish
        explore_budget = int(budget * 0.6)
        n_restarts = max(1, min(8, explore_budget // 30))
        budget_per_restart = explore_budget // n_restarts

        for r in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break

            # Diverse initialization
            if r == 0:
                x0 = rng.uniform(-4, 4, dim)
            else:
                # Mix of random and perturbed best
                if rng.random() < 0.5:
                    x0 = rng.uniform(-4, 4, dim)
                else:
                    if not np.isnan(best_x).any():
                        x0 = best_x + rng.uniform(-2, 2, dim)
                    else:
                        x0 = rng.uniform(-4, 4, dim)
            x0 = np.clip(x0, -5, 5)

            fx0 = evaluate(x0)

            # CMA-ES parameters
            pop_size = dim + 3
            sigma = 2.0
            # Covariance matrix
            C = np.eye(dim)
            # Step sizes for C and sigma
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
            cs = (dim + 1.7) / (dim + 9.2)
            cmu = min(1 - 2 / ((dim + 1.3) ** 2 + cs), 2 / ((dim + 2) ** 2 + cs))
            d_sigma = 1 + 2 * max(0, (dim - 1.7) / (dim + 9.2)) + cs

            # State
            mean = x0.copy()
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C_inv = np.eye(dim)
            # Precompute for efficiency
            sqrt_eigenvalues = np.ones(dim)

            remaining = min(budget_per_restart, budget - call_count)
            for _ in range(remaining):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                Y = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    B = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    Y[i] = mean + sigma * (B @ z)
                    Y[i] = np.clip(Y[i], -5, 5)

                # Evaluate
                F = np.array([evaluate(Y[i]) for i in range(pop_size)])

                # Sort
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]
                mu = pop_size // 2
                # Weights
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mu_eff = 1.0 / np.sum(w ** 2)

                # Update mean
                new_mean = (w[:, None] * Y_sorted[:mu]).sum(axis=0)

                # Update evolution paths
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv @ (sigma * (new_mean - mean)))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * 1)) / np.sqrt(1 - (1 - cs) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean)

                # Update covariance
                delta_h = (1 - hsig) * cc * (2 - cc)
                rank_mu = np.outer((new_mean - mean), (new_mean - mean))
                C = (1 + cs * d_sigma * (np.dot(p_sigma, p_sigma) / dim - 1) + cs * (2 - cs) - cc * (2 - cc) * delta_h + cmu) * C \
                    + cmu * (rank_mu / sigma ** 2) - (cs * d_sigma * (np.dot(p_sigma, p_sigma) / dim - 1) + cs * (2 - cs) - cc * (2 - cc) * delta_h + cmu) * C

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                sqrt_eigenvalues = np.sqrt(eigvals)

                # Update sigma
                sigma *= np.exp(cs / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                # Update mean
                mean = new_mean
                mean = np.clip(mean, -5, 5)

        # --- Phase 2: Final local refinement (Nelder-Mead style) ---
        if call_count < budget and not np.isnan(best_x).any():
            remaining = budget - call_count
            # Use Nelder-Mead for local refinement
            x_curr = best_x.copy()
            f_curr = best_f

            # Initialize simplex
            simplex = [x_curr.copy()]
            for i in range(dim):
                xi = x_curr.copy()
                xi[i] += 0.1 * (5 - x_curr[i]) if x_curr[i] < 5 else -0.1 * (x_curr[i] + 5)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex = np.array(simplex)
            f_simplex = np.array([f_curr] + [evaluate(simplex[i]) for i in range(1, dim + 1)])

            # Nelder-Mead iterations
            max_iter = min(remaining, 200)
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

                centroid = simplex[:-1].mean(axis=0)
                centroid = np.clip(centroid, -5, 5)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)

                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = evaluate(xc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = evaluate(simplex[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
