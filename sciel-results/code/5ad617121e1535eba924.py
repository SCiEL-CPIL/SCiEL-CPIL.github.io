"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive restarts on the Sphere benchmark."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        eval_count = 0

        def time_ok():
            return time.time() < deadline

        def remaining():
            return max(0, budget - eval_count)

        # Phase 1: Coarse multi-start to find a promising region
        # Use a small number of restarts with (1+1)-ES to locate basin
        num_starts = min(8, max(2, budget // 100))
        start_budget = max(20, budget // (num_starts * 2))

        rng_master = np.random.default_rng(42)

        for s in range(num_starts):
            if not time_ok() or eval_count >= budget:
                break

            # Start from a random point
            rng = np.random.default_rng(rng_master.integers(0, 2**32))
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.5
            local_best_x = x.copy()
            local_best_f = fx
            local_evals = 1

            # (1+1)-ES refinement for this start
            max_local = min(start_budget, budget - eval_count)
            for _ in range(max_local):
                if not time_ok() or eval_count >= budget:
                    break

                y = np.clip(local_best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < local_best_f:
                    local_best_x, local_best_f = y, fy
                    sigma *= 1.2
                    if local_best_f < best_f:
                        best_f, best_x = local_best_f, local_best_x.copy()
                else:
                    sigma *= 0.9

        if not time_ok() or eval_count >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES refinement around the best found point
        # This is the main optimization phase
        remaining_budget = budget - eval_count
        if remaining_budget < 10:
            return best_x, best_f

        # Initialize CMA-ES
        x0 = best_x.copy() if best_x is not None else np.zeros(dim)
        f0 = best_f if best_f < np.inf else func(x0)
        if best_x is None:
            best_x = x0.copy()

        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(dim)))  # population size
        c1 = (4 + lam / dim) / ((dim + 1.3)**2 + lam)
        cmu = min(0.4 - c1, 2 * (2 - dim / (dim + 2))**2 - c1)
        cs = (dim + 1.7) / (dim + 9.5 + lam)
        ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        sigma = 0.5  # initial step size for refinement
        xmean = x0.copy()
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        stepsizes = np.ones(dim)

        rng_cma = np.random.default_rng(123)

        # Track for restart detection
        no_improve_count = 0
        restart_threshold = 150  # restart if no improvement for this many iterations

        cma_iter = 0
        while eval_count < budget and time_ok():
            if not time_ok():
                break

            # Generate population
            z = rng_cma.standard_normal((lam, dim))

            # Apply CMA-ES transformation
            # x = xmean + sigma * B @ (D * z)
            X = np.zeros((lam, dim))
            for i in range(lam):
                X[i] = xmean + sigma * (B @ (D * z[i]))
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if eval_count >= budget or not time_ok():
                    F[i] = np.inf
                    break
                F[i] = func(X[i])
                eval_count += 1

            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            mu = lam // 2  # number of weights

            # Weights
            w = np.zeros(lam)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w**2)

            # Update mean
            xnew = np.sum(w[:, None] * X[:mu], axis=0)

            # Update step size (CSA)
            # Save old B and D
            B_old = B.copy()
            D_old = D.copy()

            # Compute evolution path
            ps = (1 - cs) * ps + cs / np.sqrt(1 - (1 - cs)**(2 * (eval_count - cma_iter * lam + 1))) * (
                np.dot(B_old.T, xnew - xmean) / (sigma * np.dot(D_old, D_old) ** 0.5) if np.dot(D_old, D_old) > 0 else np.zeros(dim)
            )

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (eval_count - cma_iter * lam + 1)))) < (1.4 + 2 / (dim + 1)) * chi

            # Update covariance matrix
            pc = (1 - c1) * pc + hsig * c1 / np.sqrt(1 - (1 - c1)**(2 * (eval_count - cma_iter * lam + 1))) * (
                np.dot(B_old.T, xnew - xmean) / (sigma * np.dot(D_old, D_old) ** 0.5) if np.dot(D_old, D_old) > 0 else np.zeros(dim)
            )

            # Rank-1 + rank-mu update
            # For simplicity, use a simplified covariance update
            # Full CMA-ES covariance update is complex; use a simplified version
            C = np.eye(dim)
            C = (1 - (c1 + cmu)) * C + c1 * np.outer(pc, pc)
            for i in range(mu):
                zi = np.dot(B_old.T, X[i] - xmean) / (sigma * np.dot(D_old, D_old) ** 0.5) if np.dot(D_old, D_old) > 0 else np.zeros(dim)
                C += cmu * w[i] * np.outer(zi, zi)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigendecomposition
            try:
                D2, V = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-12)
                D = np.sqrt(D2)
                B = V
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)

            # Update mean
            xmean = xnew

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Track best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            cma_iter += 1

            # Check for stagnation - restart CMA-ES with smaller sigma
            if no_improve_count > restart_threshold:
                no_improve_count = 0
                sigma *= 0.5
                # Reset some state
                ps = np.zeros(dim)
                pc = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
