"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, adaptive sigma, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            return val

        # --- Phase 1: Robust Initialization ---
        # Evaluate origin and several random points to find a good starting basin
        candidates = [np.zeros(dim)]
        for _ in range(9):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = safe_func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if calls_used >= budget:
            return best_x, best_f

        # --- Phase 2: CMA-ES ---
        # Population size
        lam = 4
        mu = lam // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()

        # Step size
        sigma = 2.0

        # Mean
        m = best_x.copy()

        # Covariance matrix
        C = np.eye(dim)

        # Precompute
        c_sigma = (mueff + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Inverse square root of C (for eigenvalue updates)
        # We'll use Cholesky for sampling

        def cholesky(C_mat):
            """Compute Cholesky factorization with regularization."""
            for _ in range(10):
                try:
                    L = np.linalg.cholesky(C_mat)
                    return L
                except np.linalg.LinAlgError:
                    C_mat = C_mat + 1e-6 * np.eye(dim)
            # Fallback: eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-10, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        # CMA-ES main loop
        max_cma_iters = budget - calls_used
        stagnation_count = 0
        best_cma_f = best_f

        for it in range(max_cma_iters):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Sample population
            L = cholesky(C)
            population = np.array([m + sigma * (L @ rng.standard_normal(dim)) for _ in range(lam)])
            population = np.clip(population, -5.0, 5.0)

            # Evaluate
            fvals = np.array([safe_func(x) for x in population])
            if calls_used >= budget:
                break

            # Sort
            order = np.argsort(fvals)
            population = population[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * population[:mu], axis=0)

            # Update sigma (1/5 rule)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (m - m_old))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (m - m_old) / sigma

            delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1 + 2 / dim) * np.sqrt(dim + 2 - 1) else 0

            C = (1 + c_1 * delta_h + c_mu - c_1 - c_1 * delta_h) * C \
                + c_1 * delta_h * (p_c @ p_c - C) \
                + c_mu / sigma ** 2 * np.sum(weights[:, None, None] * np.outer(population[:mu] - m_old, population[:mu] - m_old), axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-10:
                C += (1e-10 - eigvals.min()) * np.eye(dim)

            # Stagnation detection: if no improvement for a while, do a restart
            if stagnation_count > 40:
                # Do a local restart: perturb the mean
                m = m + 0.5 * sigma * rng.standard_normal(dim)
                m = np.clip(m, -5.0, 5.0)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 0.5)
                stagnation_count = 0

        # --- Phase 3: Nelder-Mead Polish ---
        # Use remaining budget for NM refinement
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline:
            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                xi = best_x.copy()
                step = 0.1 if abs(best_x[i]) > 1e-8 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)

            fvals = np.array([safe_func(x) for x in simplex])
            if calls_used < budget:
                # Nelder-Mead iterations
                max_nm_iters = remaining - 1
                for nm_it in range(max_nm_iters):
                    if time.time() > deadline:
                        break
                    if calls_used >= budget:
                        break

                    # Sort simplex
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = fvals[order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + 1.0 * (centroid - simplex[-1])
                    xr = np.clip(xr, -5.0, 5.0)
                    fr = safe_func(xr)

                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    elif fr < fvals[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, -5.0, 5.0)
                        fe = safe_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        # Contraction
                        if fr < fvals[-1]:
                            # Outside contraction
                            xc = centroid + 0.5 * (simplex[-1] - centroid)
                            xc = np.clip(xc, -5.0, 5.0)
                            fc = safe_func(xc)
                            if fc <= fr:
                                simplex[-1] = xc
                                fvals[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    fvals[i] = safe_func(simplex[i])
                        else:
                            # Inside contraction
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                            xc = np.clip(xc, -5.0, 5.0)
                            fc = safe_func(xc)
                            if fc < fvals[-1]:
                                simplex[-1] = xc
                                fvals[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    fvals[i] = safe_func(simplex[i])

                    # Update best
                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
