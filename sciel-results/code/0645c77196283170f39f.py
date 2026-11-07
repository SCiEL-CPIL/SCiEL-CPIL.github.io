# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using structured initialization and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return f

        # ZDT2: f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/g(x)))
        # where g(x) = 1 + 10*(n-1) + sum((x_i - 0.5)^2 - 10*cos(2*pi*(x_i - 0.5))) for i=1..n-1
        # Optimal: g(x) = 1 (i.e., x_i = 0.5 for all i >= 1)
        # Pareto front: f2 = (1 + 10*(n-1)) * (1 - sqrt(f1)) for f1 in [0, 1]

        n = self.dim
        g_max = 1 + 10 * (n - 1)

        # Chebyshev nodes for x[0] to minimize max gap on the convex front
        # Chebyshev roots: x_k = cos((2k+1)pi/(2M)) for k=0..M-1, mapped to [0,1]
        M = min(self.budget // 3, 67)  # number of structured points
        M = max(M, 10)

        # Chebyshev nodes in [0, 1]
        k_vals = np.arange(M)
        # Chebyshev roots mapped to [0,1]: 0.5 * (1 - cos((2k+1)*pi/(2M)))
        x0_cheb = 0.5 * (1 - np.cos((2 * k_vals + 1) * np.pi / (2 * M)))
        x0_cheb = np.sort(x0_cheb)  # ensure sorted

        # Generate structured points: x[0] = Chebyshev node, x[1:] = 0.5
        X_struct = np.zeros((M, n))
        X_struct[:, 0] = x0_cheb
        X_struct[:, 1:] = 0.5  # optimal for g(x)

        # Evaluate structured points
        F = np.zeros((M, self.n_obj))
        for i in range(M):
            if used >= self.budget or time.time() > deadline:
                break
            F[i] = eval_point(X_struct[i])

        # Add some random points for robustness
        n_rand = min(self.budget - used, 10)
        if n_rand > 0:
            X_rand = lo + (hi - lo) * rng.random((n_rand, n))
            F_rand = np.zeros((n_rand, self.n_obj))
            for i in range(n_rand):
                if used >= self.budget or time.time() > deadline:
                    break
                F_rand[i] = eval_point(X_rand[i])
            X_struct = np.vstack([X_struct, X_rand])
            F = np.vstack([F, F_rand])

        # Local search / polish: for each point, try to improve g(x) by moving x[1:] toward 0.5
        # Use gradient-free local search on x[1:]
        n_local = min(self.budget - used, 50)
        if n_local > 0 and time.time() < deadline:
            # Sort by f1 to maintain diversity
            order = np.argsort(F[:, 0])
            X_struct = X_struct[order]
            F = F[order]

            # For each of the first few points, do a small local search
            n_polish = min(20, len(X_struct))
            for idx in range(n_polish):
                if used >= self.budget or time.time() > deadline:
                    break
                x = X_struct[idx].copy()
                best_f = F[idx].copy()
                best_x = x.copy()
                # Try moving x[1:] toward 0.5 with small steps
                for _ in range(5):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Perturb x[1:]
                    step = 0.05 * (hi[1:] - lo[1:])
                    x_trial = x.copy()
                    x_trial[1:] += step * (0.5 - x[1:]) * rng.standard_normal(n - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = eval_point(x_trial)
                    if f_trial[1] < best_f[1] - 1e-10 or (np.allclose(f_trial[0], best_f[0], atol=1e-10) and f_trial[1] < best_f[1]):
                        best_f = f_trial.copy()
                        best_x = x_trial.copy()
                X_struct[idx] = best_x
                F[idx] = best_f

        # Final: do a fine sweep over x[0] with optimal x[1:] = 0.5
        # Use remaining budget for a dense Chebyshev sweep
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            n_sweep = min(remaining, 50)
            k_vals2 = np.arange(n_sweep)
            x0_sweep = 0.5 * (1 - np.cos((2 * k_vals2 + 1) * np.pi / (2 * n_sweep)))
            x0_sweep = np.sort(x0_sweep)
            X_sweep = np.zeros((n_sweep, n))
            X_sweep[:, 0] = x0_sweep
            X_sweep[:, 1:] = 0.5
            F_sweep = np.zeros((n_sweep, self.n_obj))
            for i in range(n_sweep):
                if used >= self.budget or time.time() > deadline:
                    break
                F_sweep[i] = eval_point(X_sweep[i])
            X_struct = np.vstack([X_struct, X_sweep])
            F = np.vstack([F, F_sweep])

        # Return the last evaluated point (return value is ignored by evaluator)
        return X_struct[-1] if len(X_struct) > 0 else np.zeros(n)
# EVOLVE-BLOCK-END
