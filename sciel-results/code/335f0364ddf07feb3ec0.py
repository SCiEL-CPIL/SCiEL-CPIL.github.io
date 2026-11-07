"""Optimizer using CMA-ES with restarts for Schaffers F7 ill-conditioned function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """CMA-ES based optimizer with restarts for multimodal/ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x = np.asarray(x, dtype=np.float64)
                val = func(x)
                call_count += 1
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                call_count += 1
                return np.inf

        best_x = np.zeros(d)
        best_f = np.inf

        # Phase 1: Broad global search with multiple random restarts of CMA-ES
        # Use a deterministic set of starting points for reproducibility
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(d))
        # Random points in [-4, 4]^d
        for _ in range(15):
            starts.append(rng.uniform(-4.0, 4.0, d))
        # Some points near the boundary
        for _ in range(5):
            starts.append(rng.uniform(-4.5, 4.5, d))

        # Allocate budget: use ~60% for global exploration, ~40% for local refinement
        global_budget = int(budget * 0.55)
        local_budget = budget - global_budget

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))
        sigma_init = 2.0

        # Try multiple starting points with CMA-ES
        for start_idx, x0 in enumerate(starts):
            if call_count >= global_budget or time.time() > deadline:
                break

            remaining_budget = global_budget - call_count
            if remaining_budget < pop_size * 3:
                break

            # Run CMA-ES from this starting point
            result = self._cma_es(func, safe_call, x0, pop_size, sigma_init,
                                  remaining_budget, d, deadline)
            if result is not None:
                x_best, f_best = result
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

            # Check if we're making good progress; if not, try another start
            if best_f < 1e-3:
                break

        # Phase 2: Local refinement using Nelder-Mead from the best point
        if call_count < budget and best_x is not None:
            local_budget_remaining = budget - call_count
            if local_budget_remaining > 10:
                nm_result = self._nelder_mead(func, safe_call, best_x, local_budget_remaining,
                                               d, deadline)
                if nm_result is not None:
                    x_best, f_best = nm_result
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()

        # Phase 3: Fine-tuning with very small perturbations
        if call_count < budget and best_x is not None:
            for sigma in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
                if call_count >= budget or time.time() > deadline:
                    break
                for _ in range(20):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x = np.clip(best_x + sigma * rng.standard_normal(d), -5, 5)
                    f = safe_call(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

        return best_x, best_f

    def _cma_es(self, func, safe_call, x0, pop_size, sigma, max_evals, d, deadline):
        """Run CMA-ES with given parameters."""
        x = x0.copy().astype(np.float64)
        sigma = float(sigma)
        pop = pop_size
        lambda_ = pop

        # CMA-ES parameters
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / lambda_) / (lambda_ + 4 + 2 * mu_eff / lambda_)
        cs = (mu_eff + 2) / (lambda_ + mu_eff + 5)
        c1 = (2 + 2 / (mu_eff - 1)) / ((lambda_ + 2) ** 2 - 2)
        cmu = min(1 - c1, (2 + (2 / (mu_eff - 1) - 2)) / ((lambda_ + 2) ** 2 - 2))

        # Initialize
        C = np.eye(d)
        C_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        best_x = x.copy()
        best_f = np.inf
        eval_count = 0

        for gen in range(1000):
            if eval_count >= max_evals or time.time() > deadline:
                break

            # Generate population
            X = np.zeros((lambda_, d))
            F = np.zeros(lambda_)

            for i in range(lambda_):
                if eval_count >= max_evals or time.time() > deadline:
                    break
                z = np.random.standard_normal(d)
                xi = x + sigma * C_sqrt @ z
                xi = np.clip(xi, -5, 5)
                X[i] = xi
                F[i] = safe_call(xi)
                eval_count += 1

            if eval_count >= max_evals or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()

            # Mean update
            m_new = np.sum(weights[:, None] * X[:mu], axis=0)

            # Covariance matrix adaptation
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            diff = (m_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (eval_count + 1)) <
                    (1 + 2 / (mu_eff + 1)) * chi_n)
            ps = ps if hsig else ps
            pc = (1 - cc) * pc + hsig * cc * diff
            C = ((1 - c1 - cmu) * C +
                 c1 * (pc @ pc.T + (1 - hsig) * cc * (1 - cc) * C) +
                 cmu * np.sum(weights[i] * np.outer(X[i] - x, X[i] - x) / sigma ** 2
                              for i in range(mu)))

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            C_sqrt, _ = np.linalg.sqrtm(C)
            # Handle potential complex from sqrtm
            if np.iscomplexobj(C_sqrt):
                C_sqrt = C_sqrt.real

            x = m_new

            # Step size adaptation
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check for convergence
            if np.linalg.norm(m_new - x) < 1e-10:
                break

        return best_x, best_f

    def _nelder_mead(self, func, safe_call, x0, max_evals, d, deadline):
        """Run Nelder-Mead optimization."""
        try:
            # Initialize simplex
            simplex = [x0.copy() + 1e-4 * np.random.standard_normal(d) for _ in range(d + 1)]
            simplex[0] = x0.copy()

            fvals = [safe_call(s) for s in simplex]
            eval_count = d + 1

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 2.0

            for _ in range(1000):
                if eval_count >= max_evals or time.time() > deadline:
                    break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_call(x_r)
                eval_count += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_call(x_e)
                    eval_count += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_call(x_c)
                    eval_count += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_call(simplex[i])
                            eval_count += 1

            # Return best
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]
        except Exception:
            return None
# EVOLVE-BLOCK-END
