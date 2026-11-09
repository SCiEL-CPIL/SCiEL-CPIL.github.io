"""Rewritten optimizer: CMA-ES with restarts and time guard."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return float(func(x))

        # Use multiple restarts with different initializations
        # Strategy: several CMA-ES runs with different seeds/centers
        max_restarts = 5
        calls_per_restart = budget // max_restarts

        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Different center for each restart
            if restart == 0:
                center = np.zeros(dim)
            elif restart == 1:
                center = np.array([2.0] * dim)
            elif restart == 2:
                center = np.array([-2.0] * dim)
            elif restart == 3:
                center = np.array([3.0, -3.0, 2.0, -2.0, 3.0])
            else:
                center = np.array([-3.0, 3.0, -2.0, 2.0, -3.0])

            # CMA-ES parameters
            pop_size = max(4 + int(3 * np.log(dim)), 6)
            # Adjust population size based on remaining budget
            if calls_per_restart < pop_size * 5:
                pop_size = max(4, calls_per_restart // 5)

            # Initial sigma
            sigma = 2.0

            # CMA-ES state
            mean = center.copy().astype(float)
            # Clip mean to box
            mean = np.clip(mean, -5, 5)

            # Covariance matrix (identity initially)
            C = np.eye(dim)
            # Step sizes
            cs = (pop_size + 2) / (dim + pop_size + 5)
            ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
            cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
            cmu = min(0.1, (pop_size - 2 + 1. / pop_size) / ((dim + 2) * (dim + pop_size)))
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Evaluate initial mean
            if calls_made < budget:
                fmean = safe_func(mean)
                if fmean < global_best_f:
                    global_best_f = fmean
                    global_best_x = mean.copy()

            # Main CMA-ES loop
            for gen in range(1000):
                if calls_made + pop_size > budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                # Eigendecomposition of C
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))

                # Sample
                ys = []
                xs = []
                for i in range(pop_size):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    z = np.random.standard_normal(dim)
                    y = sigma * (B @ (D * z))
                    x = mean + y
                    x = np.clip(x, -5, 5)
                    ys.append(y)
                    xs.append(x)

                if len(xs) < 2:
                    break

                # Evaluate
                fs = []
                for x in xs:
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fx = safe_func(x)
                    fs.append(fx)
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()

                if len(fs) < 2:
                    break

                # Sort
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                xs = [xs[i] for i in idx]
                fs = [fs[i] for i in idx]

                # Update ps
                w = np.log(pop_size / 2 + 0.5) - np.arange(1, pop_size + 1)
                w = w / np.sum(w)
                mu_eff = 1 / np.sum(w[:pop_size // 2] ** 2)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D * np.mean(ys[:pop_size // 2], axis=0)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chiN

                # Update pc
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D * np.mean(ys[:pop_size // 2], axis=0)))

                # Update mean
                mean = mean + np.sum([w[i] * ys[i] for i in range(min(pop_size // 2, len(ys)))], axis=0) if len(ys) > 0 else mean
                mean = np.clip(mean, -5, 5)

                # Update C
                if len(ys) >= 2:
                    delta_mean = np.mean(ys[:pop_size // 2], axis=0)
                    C = (1 + cmu * mu_eff - cc) * C + \
                        cmu * np.outer(np.sum([w[i] * ys[i] for i in range(min(pop_size // 2, len(ys)))], axis=0),
                                       np.sum([w[i] * ys[i] for i in range(min(pop_size // 2, len(ys)))], axis=0)) + \
                        (1 + cmu * mu_eff - cc - ds) * np.outer(pc, pc)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-8, 10)

                # Convergence check
                if np.max(np.abs(D)) < 1e-8 * (1 + np.linalg.norm(mean)):
                    break

        return global_best_x, global_best_f
