"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es_run(rng, x0, sigma0, budget_left):
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            lam = 10  # population size
            mu = lam // 2  # number of selected individuals

            # CMA-ES parameters
            cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
            cs = (n + 1.3) / (n + 3 + mu_eff)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            # Evaluate initial point
            fx = func(x)
            track_call(x, fx)

            gen = 0
            while calls_used < total_budget and (total_budget - calls_used) >= lam:
                if time.time() > deadline:
                    break
                gen += 1

                # Generate population
                Y = []
                F = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    track_call(y, fy)
                    if calls_used >= total_budget:
                        break
                    Y.append(y)
                    F.append(fy)

                if len(Y) < lam:
                    break

                # Sort by fitness
                order = np.argsort(F)
                Y = [Y[i] for i in order]
                F = [F[i] for i in order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Selection weights
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights ** 2)

                # Mean update
                y_w = sum(weights[i] * Y[i] for i in range(mu))
                x_old = x.copy()
                x = y_w

                # Step-size control
                z_c = (B @ (D * rng.standard_normal(n)))  # dummy for direction
                # Actually compute the normalized step
                # The actual step in C space: sum w_i * z_i
                # We need to track the z vectors
                # Let's redo: store z vectors
                # For simplicity, recompute
                # Actually, let's just use the standard update
                # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * (B.T @ (D * ((y_w - x_old)/sigma)))
                # But we need the z vectors. Let me restructure.
                
                # For now, use a simplified but effective approach
                # Update sigma based on success rate
                success_rate = sum(1 for i in range(mu) if F[i] < fx) / mu
                if success_rate > 0.5:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 1e-6, 10.0)

                # Update covariance (simplified: use PCA of selected points)
                if mu >= 2:
                    selected = np.array(Y[:mu])
                    # Center
                    mean_sel = np.mean(selected, axis=0)
                    centered = selected - mean_sel
                    # Covariance
                    C_new = (centered.T @ centered) / (mu - 1)
                    # Blend
                    C = 0.9 * C + 0.1 * C_new
                    # Ensure positive definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    # Update B and D
                    eigvals2, B_new = np.linalg.eigh(C)
                    eigvals2 = np.clip(eigvals2, 1e-10, None)
                    D = np.sqrt(eigvals2)
                    B = B_new

                fx = F[0]

        # Phase 1: Brief random exploration
        rng = np.random.default_rng(42)
        explore_budget = max(5, total_budget // 20)
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            if calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        # Phase 2: CMA-ES from best point
        remaining = total_budget - calls_used
        if remaining > lam + 10:
            cma_es_run(rng, best_x, 1.0, remaining)

        # Phase 3: If budget remains, try another CMA-ES from a different point
        remaining = total_budget - calls_used
        if remaining > lam + 10:
            x2 = rng.uniform(-5, 5, dim)
            cma_es_run(rng, x2, 2.0, remaining)

        # Phase 4: Final local refinement with small sigma CMA-ES
        remaining = total_budget - calls_used
        if remaining > lam + 10:
            cma_es_run(rng, best_x, 0.1, remaining)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
