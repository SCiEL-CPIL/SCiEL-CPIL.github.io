"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        self._best_x = best_x
        self._best_f = best_f

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, sigma0, budget_left, max_iter=300):
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs

            mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            rng = np.random.default_rng()

            # Generate initial population
            pop = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = mean + sigma * (B @ (D * z))
                pop_f[i] = eval_func(pop[i])

            it = 0
            while calls < budget_left and it < max_iter:
                if time.time() > deadline:
                    break
                it += 1

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update mean
                old_mean = mean.copy()
                mean = np.dot(w[:mu], pop[:mu])

                # Update step size control variables
                y = (mean - old_mean) / sigma
                # Transform to principal axes
                y_trans = B.T @ (D * y)  # Actually we need: (1/D) * B.T @ y
                # Correct: ps update uses the transformed y
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (B.T @ (D * y) / D)
                # Simplified: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * B.T@(y)  -- but y is in original space
                # Let me use the standard form:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) / sigma * B.T @ (mean - old_mean)
                # Wait, the standard CMA-ES uses:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (1/sigma) * B^T * D^(-1) * (mean - old_mean)
                # Actually the correct formula is:
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (mean - old_mean) / (sigma * D)  in B-space
                # Let me just use the standard implementation:
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (B.T @ ((mean - old_mean) / sigma))

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (it + 1))) < \
                       (1.4 + 2.0 / (n + 1.0)) * chi_n

                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * \
                     (B.T @ ((mean - old_mean) / sigma))

                # Update covariance matrix
                delta_d = cc * hsig + (1.0 - cc)
                delta_d = min(delta_d, 1.0)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (pop[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(B.T @ (D * y_i), B.T @ (D * y_i))
                    # Actually: rank_mu += w[i] * outer((pop[i]-old_mean)/sigma, (pop[i]-old_mean)/sigma)
                    # But in the transformed space:
                # Let me redo this more carefully:
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = (pop[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(diff, diff)

                C = (1.0 + c1 * delta_d - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Re-orthonormalize using eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    # Fallback: reset C
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 1e3)

                # Generate new population
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = mean + sigma * (B @ (D * z))
                    pop_f[i] = eval_func(pop[i])

                # Convergence check
                if np.linalg.norm(mean - old_mean) / max(sigma, 1e-15) < 1e-10:
                    break

                # If sigma is very small, we've converged
                if sigma < 1e-10:
                    break

            return mean, sigma, it

        # Main optimization loop with restarts
        rng = np.random.default_rng(0)

        # Strategy: Multiple CMA-ES restarts from different starting points
        # f19 is multimodal, so restarts help escape local optima

        # Phase 1: Initial exploration with a few random starts
        # Use 3 restarts, each with a portion of the budget
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts

        for restart in range(n_restarts):
            if calls >= self.budget or time.time() > deadline:
                break

            if restart == 0:
                # First restart: start from origin
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
            elif restart == 1:
                # Second restart: start from current best, perturbed
                x0 = best_x + rng.standard_normal(self.dim) * 0.5
                x0 = np.clip(x0, -self.bound, self.bound)
                sigma0 = 1.0
            else:
                # Third restart: random start
                x0 = rng.uniform(-3.0, 3.0, self.dim)
                sigma0 = 1.5

            remaining = self.budget - calls
            x0, sigma0, _ = cma_es(x0, sigma0, remaining)

        # Phase 2: Final local refinement with L-BFGS-B
        if calls < self.budget and time.time() < deadline:
            from scipy.optimize import minimize

            def wrapped_func(x):
                if calls >= self.budget or time.time() > deadline:
                    return np.inf
                x = np.clip(x, -self.bound, self.bound)
                try:
                    return float(func(x))
                except Exception:
                    return np.inf

            bounds = [(-self.bound, self.bound)] * self.dim
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 30}
                )
                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -self.bound, self.bound)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
