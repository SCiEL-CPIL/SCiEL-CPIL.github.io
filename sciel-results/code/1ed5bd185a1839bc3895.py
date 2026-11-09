# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES-like optimizer with time guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Time-aware call wrapper
        calls_made = [0]

        def timed_call(x):
            if time.time() > deadline:
                raise TimeoutError()
            calls_made[0] += 1
            return func(x)

        # Initialize with a few random starts to find a good basin
        rng = np.random.default_rng(42)

        # Try several initial points
        best_x = None
        best_f = np.inf
        for _ in range(5):
            x0 = rng.uniform(-4, 4, n)
            fx = timed_call(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * n  # population size
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = mu / (weights ** 2).sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        sigma = 2.0
        x_mean = best_x.copy()
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        # Precompute eigendecomposition for C update (lazy)
        D = np.ones(n)  # diagonal of C^{-1/2}

        def matvec_C_inv_sqrt(v):
            # C^{-1/2} v = D^{-1} (eigvecs^T v) ... we'll use Cholesky-like update
            # For simplicity, use direct matrix inverse sqrt via eigen
            # But that's expensive; use the standard CMA-ES update formula
            # Actually, let's just use np.linalg.solve with C
            # For efficiency, we'll maintain C and use its inverse
            C_inv = np.linalg.inv(C)
            return np.linalg.solve(np.linalg.cholesky(C_inv).T,
                                   np.linalg.cholesky(C_inv) @ v)

        # Better: maintain C and compute C^{-1/2} via eigen decomposition periodically
        # For n=5, this is cheap
        def get_C_inv_sqrt():
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            D_inv = 1.0 / np.sqrt(eigvals)
            return (eigvecs * D_inv) @ eigvecs.T

        C_inv_sqrt = get_C_inv_sqrt()

        best_x = x_mean.copy()
        best_f = best_f

        while calls_made[0] < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                pop.append((y, i))

            # Evaluate
            evals = []
            for y, idx in pop:
                if time.time() > deadline - 2.0:
                    break
                fy = timed_call(y)
                evals.append((fy, y, idx))

            if not evals:
                break

            # Sort by fitness
            evals.sort(key=lambda t: t[0])

            # Update best
            for fy, y, idx in evals:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Select top mu
            selected = evals[:mu]
            y_w = np.array([s[1] for s in selected])

            # Update mean
            y_w_mean = np.average(y_w, axis=0, weights=weights)
            x_mean = np.clip(x_mean + (y_w_mean - x_mean), -5, 5)

            # Update step size
            y_diff = y_w - x_mean
            # Compute ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (y_w_mean - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made[0] // lambda_)))) < \
                   (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (calls_made[0] // lambda_)))

            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_w_mean - x_mean) / sigma

            # Update C
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu / sigma ** 2 * np.sum(weights[i] * np.outer(y_w[i] - x_mean, y_w[i] - x_mean)
                                           for i in range(mu))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += 1e-12 * np.eye(n)

            # Recompute C^{-1/2}
            C_inv_sqrt = get_C_inv_sqrt()

            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Safety: if sigma gets too small or too large, restart
            if sigma < 1e-8 or sigma > 100.0:
                # Restart with new random point
                x0 = rng.uniform(-4, 4, n)
                fx = timed_call(x0)
                if fx < best_f:
                    best_f = fx
                    best_x = x0.copy()
                x_mean = best_x.copy()
                sigma = 2.0
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
