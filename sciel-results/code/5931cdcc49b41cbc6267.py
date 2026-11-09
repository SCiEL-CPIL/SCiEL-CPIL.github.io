"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with time-aware budget allocation.
    Combines the structural robustness of CMA-ES (inspiration 2's use of a
    sophisticated optimizer) with the tight deadline guarding and simple
    step-size adaptation logic of the current (1+1)-ES (current program).
    
    Strategy:
    1. Use a small initial random search to find a good starting point.
    2. Run CMA-ES from that point, allocating most of the budget to it.
    3. If time/budget remains, run a second CMA-ES restart from the best
       point found, or a (1+1)-ES polish if very little budget remains.
    
    CMA-ES is chosen because it handles the Rosenbrock valley (f8) much
    better than simple (1+1)-ES, especially when the shift/rotation is
    unknown. It adapts its covariance matrix to the local landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        # 300s limit, keep 20s safety margin for evaluation overhead
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        n = self.dim
        b = self.budget

        # Track best solution found so far
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= b:
                return np.inf
            if time.perf_counter() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Initial random search to find a good starting point ---
        # Use ~10% of budget for initial exploration
        n_init = max(10, b // 10)
        for _ in range(n_init):
            if call_count >= b or time.perf_counter() > deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, n)
            safe_call(x0)

        if best_x is None:
            # Fallback: should not happen if b > 0
            best_x = np.zeros(n)
            best_f = safe_call(best_x)

        # --- Phase 2: CMA-ES from the best initial point ---
        # Allocate remaining budget to CMA-ES
        remaining_budget = b - call_count
        if remaining_budget > 20 and time.perf_counter() < deadline:
            # CMA-ES implementation
            x = best_x.copy()
            f = best_f
            sigma = 1.0  # Initial step size
            C = np.eye(n)  # Covariance matrix
            C_inv = np.eye(n)
            ps = np.zeros(n)  # Evolution path for sigma
            pc = np.zeros(n)  # Evolution path for C
            # Learning rates
            mu = max(1, n // 2)  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / sum(weights**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / (n**2 + 1)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (n**2 + 1))

            # Limit CMA-ES iterations to avoid infinite loops
            max_iter = min(remaining_budget // 2, 500)  # CMA-ES uses ~n+1 calls per iter
            for it in range(max_iter):
                if call_count >= b or time.perf_counter() > deadline:
                    break

                # Generate lambda candidates
                lam = 2 * mu  # Population size
                ys = []
                for j in range(lam):
                    if call_count >= b or time.perf_counter() > deadline:
                        break
                    # Generate candidate
                    y = x + sigma * (C @ rng.standard_normal(n))
                    fy = safe_call(y)
                    ys.append((fy, y))

                if len(ys) < mu:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[0])

                # Update evolution paths
                y_mean = sum(weights[j] * ys[j][1] for j in range(mu))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv @ (y_mean / sigma)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))):
                    pc = (1 - c_c) * pc + c_c * (y_mean / sigma)
                    c_1 = 2 / (n**2 + 1)
                    c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (n**2 + 1))
                    C = (1 + c_1 + c_mu - c_1 - c_c * (1 - c_c)) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * sum(weights[j] * np.outer(ys[j][1] - x, ys[j][1] - x) / sigma**2 for j in range(mu))
                    # Symmetrize
                    C = (C + C.T) / 2
                    # Ensure positive definite
                    try:
                        C_inv = np.linalg.inv(C)
                    except np.linalg.LinAlgError:
                        C_inv = np.eye(n)
                        C = np.eye(n)

                # Update mean
                x = y_mean
                f = ys[0][0]

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) - np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))))
                sigma = np.clip(sigma, 1e-10, 10.0)

        # --- Phase 3: (1+1)-ES polish if budget remains ---
        remaining_budget = b - call_count
        if remaining_budget > 5 and time.perf_counter() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Smaller step size for polishing
            for _ in range(remaining_budget):
                if time.perf_counter() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-8)

        # Return best solution found
        if best_x is None:
            best_x = np.zeros(n)
            best_f = np.inf
        return best_x, best_f
# EVOLVE-BLOCK-END
