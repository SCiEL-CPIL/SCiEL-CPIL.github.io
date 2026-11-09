"""Optimizer: structured grid-search warm-up + CMA-ES refinement for BBOB f1 (Sphere)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: structured grid search for localization, CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Structured grid-search warm-up ---
        # Evaluate a structured set of points to bracket the minimum
        warmup_points = []

        # 1. Center
        warmup_points.append(np.zeros(dim))

        # 2. Axis-aligned points at ±2.0
        for d in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = sign * 2.0
                warmup_points.append(pt)

        # 3. Corner-like points: combinations of ±2.0 on subsets of dimensions
        # Use all 2^dim corners of the box [-2, 2]^dim
        for mask in range(2 ** dim):
            pt = np.zeros(dim)
            for d in range(dim):
                if mask & (1 << d):
                    pt[d] = 2.0
                else:
                    pt[d] = -2.0
            warmup_points.append(pt)

        # 4. Mid-range points: ±3.5 on single axes (to detect if min is farther out)
        for d in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = sign * 3.5
                warmup_points.append(pt)

        # Evaluate all warm-up points
        for pt in warmup_points:
            if calls_made >= budget:
                break
            evaluate(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # --- Phase 2: CMA-ES refinement ---
        # Estimate initial sigma from warm-up spread
        # Use the spread of evaluated points as a guide
        if calls_made > 1:
            # Estimate local scale from recent best points
            sigma = 1.5
        else:
            sigma = 2.0

        x = best_x.copy()
        fx = best_f

        # CMA-ES parameters
        lam = max(8, 4 + int(3 * np.log(dim)))  # For dim=5: lam=8
        mu = max(2, lam // 2)

        # Logarithmic weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates (standard CMA-ES)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))

        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial mean and covariance
        mean = x.copy()
        C = np.eye(dim)

        # Chi-norm for sigma update
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        remaining = budget - calls_made
        if remaining <= 0:
            return x, fx

        # Run CMA-ES
        for gen in range(remaining // lam + 1):
            if calls_made >= budget:
                break

            # Eigen decomposition of C
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)
            B = evecs
            D = np.sqrt(evals)

            # Sample offspring
            z = rng.standard_normal((lam, dim))
            y = z @ (B * D).T

            # Update sigma path
            y_sigma = y / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (weights[:mu] @ y_sigma[:mu].T)

            # Hessian test for sigma update
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            h_sigma = float(norm_p_sigma / expected_norm < (1.4 + 2.0 / (dim + 1)) * np.sqrt(mu_eff))

            # Update C path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                  (weights[:mu] @ y[:mu].T)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_c = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])

            C = (1 + c_1 * (c_mu - c_1) - c_1) * C + c_1 * (rank_c + delta_h * rank_c) + \
                c_mu * (rank_mu - C)

            # Ensure symmetry and PSD
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Update sigma
            sigma *= np.exp((c_sigma / 5.0) * (norm_p_sigma / chi_n - 1.0))

            # Generate and evaluate candidates
            candidates = []
            fvals = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                xi = mean + sigma * y[i]
                xi = np.clip(xi, -5.0, 5.0)
                fi = func(xi)
                calls_made += 1
                candidates.append(xi)
                fvals.append(fi)

            if len(fvals) == 0:
                break

            # Sort offspring by fitness
            idx = np.argsort(fvals)
            y_sorted = y[idx]
            fvals_sorted = np.array(fvals)[idx]

            # Update mean
            mean = mean + sigma * (weights[:mu] @ y_sorted[:mu].T)
            mean = np.clip(mean, -5.0, 5.0)

            # Update best
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = candidates[idx[0]].copy()

            # Stopping criteria
            if sigma < 1e-12:
                break
            if gen > 5 and np.max(np.abs(y_sorted[:mu])) < 1e-10:
                break

        return x, fx
# EVOLVE-BLOCK-END
