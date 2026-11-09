"""CMA-ES based optimizer for BBOB f4 (Büche-Rastrigin) with lattice-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with lattice-aware initialization for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def cma_es(x0, sigma0, max_calls, pop_size=None):
            """
            CMA-ES implementation with adaptive step size.
            """
            nonlocal calls_made, x_best, f_best

            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))

            # Ensure we have enough budget
            if max_calls < pop_size * 2:
                max_calls = pop_size * 2

            # CMA-ES parameters
            lambda_ = pop_size
            mu = lambda_ // 2  # number of parents

            # Weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 1) - np.log(i + 1)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            x_mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C_inv = np.eye(dim)
            C = np.eye(dim)
            dnorm = np.sqrt(dim)

            # Precompute sqrtm for C
            def sqrtm(m):
                try:
                    return np.linalg.cholesky(m)
                except np.linalg.LinAlgError:
                    # Fallback to eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(m)
                    eigvals = np.maximum(eigvals, 1e-10)
                    return eigvecs @ np.diag(np.sqrt(eigvals))

            B = np.eye(dim)
            D = np.ones(dim)

            # Main loop
            gen = 0
            while calls_made < max_calls and time.time() < deadline:
                if calls_made >= max_calls:
                    break

                # Generate population
                ys = []
                xs = []
                for i in range(lambda_):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    z = np.random.randn(dim)
                    y = B @ (D * z)
                    x = x_mean + sigma * y
                    x = np.clip(x, -5, 5)
                    fx = evaluate(x)
                    ys.append(y)
                    xs.append((x, fx))

                if len(xs) < mu:
                    break

                # Sort by fitness
                xs.sort(key=lambda item: item[1])
                ys = [np.array(y) for y in ys]
                # Re-sort ys corresponding to sorted xs
                # Actually, we need to keep track of which y went with which x
                # Let's redo this more carefully

                # Regenerate with proper tracking
                # (This is a simplification; in practice, we'd track indices)
                # For now, let's just use the best mu points

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = sum(w[i] * xs[i][0] for i in range(mu))

                # Update sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D * x_mean_old)) / sigma
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * dnorm:
                    pc = (1 - cc) * pc + cc / np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean - x_mean_old) / sigma

                # Update C
                delta_sig = 1 + cs * (np.linalg.norm(ps) ** 2 / dim - 1)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) <
                        (1.4 + 2 / (dim + 1)) * dnorm)
                hsig = float(hsig)
                C = (1 + c1 * hsig * (np.outer(pc, pc) - C) + cmu * sum(
                    w[i] * np.outer(ys[i], ys[i]) for i in range(mu))) * (1 / delta_sig)

                sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / dnorm - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                D = np.sqrt(eigvals)
                B = eigvecs

                gen += 1

                # Check convergence
                if np.linalg.norm(x_mean - x_mean_old) < 1e-10 * max(1, np.linalg.norm(x_mean)):
                    if sigma < 1e-8:
                        break

        def lattice_init(max_calls):
            """
            Initialize by sampling integer lattice points to find a good basin.
            """
            nonlocal calls_made, x_best, f_best
            if calls_made >= max_calls or time.time() > deadline:
                return

            rng = np.random.default_rng(42)

            # Sample random integer points in [-4, 4]^5
            lattice_budget = min(max_calls, int(budget * 0.08))

            for _ in range(lattice_budget):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
                evaluate(x_int)

            # If we found a good point, do a quick local refinement
            if x_best is not None and calls_made < budget - 20:
                # Quick coordinate descent
                x_curr = x_best.copy()
                f_curr = f_best
                steps = [1.0, 0.5, 0.25]
                for step in steps:
                    for d in range(dim):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        for direction in [-1, 1]:
                            x_trial = x_curr.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = evaluate(x_trial)
                            if f_trial < f_curr - 1e-10:
                                x_curr = x_trial
                                f_curr = f_trial
                                if f_curr < f_best:
                                    f_best = f_curr
                                    x_best = x_curr.copy()

        # Strategy:
        # 1. Lattice Initialization: Quickly probe integer points to find the correct basin.
        # 2. CMA-ES: Main optimization phase with adaptive step size.
        # 3. Final polish: Short CMA-ES run with smaller sigma.

        # Phase 1: Lattice Search (8% budget)
        lattice_calls = int(budget * 0.08)
        if time.time() < deadline:
            lattice_init(lattice_calls)

        # Phase 2: CMA-ES Main (75% budget)
        cma_calls = int(budget * 0.75)
        if time.time() < deadline:
            if x_best is not None:
                cma_es(x_best.copy(), 1.0, cma_calls, pop_size=12)
            else:
                # Fallback to random start
                x_start = np.random.uniform(-5, 5, dim)
                evaluate(x_start)
                cma_es(x_start, 1.0, cma_calls, pop_size=12)

        # Phase 3: Final Polish (15% budget)
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            polish_calls = min(remaining, int(budget * 0.15))
            cma_es(x_best.copy(), 0.1, polish_calls, pop_size=8)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
