"""CMA-ES with multi-start and adaptive restart for BBOB Weierstrass f16.

Structural design:
  - Primary engine: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    which adapts the full covariance structure, giving it strong performance
    on rotated/shifted landscapes like BBOB instances.
  - Multi-start: several initial CMA-ES runs from diverse starting points to
    handle the hidden shift and rotation.
  - Adaptive restart: when a CMA-ES run stagnates, restart with a new center
    and reset the covariance, preserving the best solution found.
  - Final polish: a short CMA-ES run with very small sigma for precision.

The key structural change from the parent is replacing the (1+1)-ES + Nelder-Mead
hybrid with a proper CMA-ES implementation, which is the gold standard for
black-box continuous optimization on BBOB-style problems.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with multi-start for BBOB Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 270.0
        n_calls = [0]

        def safe(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float, copy=True)

        # ---- CMA-ES implementation -------------------------------------------
        def cma_es(start_x, sigma0, max_evals):
            """Run CMA-ES from start_x with initial step size sigma0."""
            nonlocal best_x, best_f

            if n_calls[0] >= budget or time.time() > deadline:
                return

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lam // 2  # number of selected individuals
            # Weights
            weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).mean()
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

            # State
            x_mean = np.array(start_x, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Initial population
            pop_x = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * z
                x = np.clip(x, -5.0, 5.0)
                f = safe(x)
                pop_x.append(x)
                pop_f.append(f)
                if np.isfinite(f) and f < best_f:
                    record(x, f)

            if n_calls[0] >= budget or time.time() > deadline:
                return

            # Main loop
            while n_calls[0] < budget and time.time() < deadline:
                # Sort by fitness
                order = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = sum(w * pop_x[i] for i, w in enumerate(weights))

                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * C @ (x_mean - x_mean_old) / sigma
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chi_n
                cc_eff = cc * (1 + (1 - hsig) * cs * (1 - cs) if hsig else 0)
                pc = (1 - cc) * pc + cc_eff * hsig * (x_mean - x_mean_old) / sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)

                # Update covariance
                delta_sig = np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1)) - 1
                rank_one = np.outer(pc, pc)
                rank_mu = sum(weights[i] * np.outer(pop_x[i] - x_mean_old, pop_x[i] - x_mean_old) for i in range(mu))
                C = (1 + c1 * delta_sig**2 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Ensure symmetry
                C = 0.5 * (C + C.T)

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(lam):
                    # Cholesky decomposition for sampling
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset
                        C = np.eye(dim)
                        L = np.eye(dim)
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * L @ z
                    x = np.clip(x, -5.0, 5.0)
                    f = safe(x)
                    new_pop_x.append(x)
                    new_pop_f.append(f)
                    if np.isfinite(f) and f < best_f:
                        record(x, f)

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Stagnation check: if no improvement in recent evaluations
                if len(pop_f) > 0 and np.isfinite(min(pop_f)):
                    if min(pop_f) >= best_f - 1e-15:
                        # Check if sigma is very small
                        if sigma < 1e-6:
                            break

                # Check for convergence
                if sigma < 1e-8:
                    break

                # Time check
                if time.time() > deadline:
                    break

        # ---- Multi-start strategy ---------------------------------------------
        # Start from several diverse points
        starts = [np.zeros(dim)]
        # Random starts
        for _ in range(5):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        # Structured starts
        for sign in [1, -1]:
            starts.append(np.full(dim, sign * 2.0))
            starts.append(np.full(dim, sign * 3.5))

        # Evaluate all starts first
        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            f = safe(s)
            if np.isfinite(f):
                record(s, f)

        # Run CMA-ES from the best start
        if n_calls[0] < budget and time.time() < deadline:
            # Use a fraction of budget for the first CMA-ES run
            first_budget = int((budget - n_calls[0]) * 0.5)
            cma_es(best_x, 2.0, first_budget)

        # Second CMA-ES run from a different start if budget remains
        if n_calls[0] < budget and time.time() < deadline:
            # Try from a random point
            start2 = rng.uniform(-3.0, 3.0, dim)
            f2 = safe(start2)
            if np.isfinite(f2) and f2 < best_f:
                best_f = f2
                best_x = start2.copy()
            second_budget = int((budget - n_calls[0]) * 0.4)
            cma_es(best_x, 1.0, second_budget)

        # Final polish with very small sigma
        if n_calls[0] < budget and time.time() < deadline:
            polish_budget = budget - n_calls[0]
            if polish_budget > 10:
                cma_es(best_x, 0.01, polish_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
