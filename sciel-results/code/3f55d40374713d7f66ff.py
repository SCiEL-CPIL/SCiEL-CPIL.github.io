"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard for 500-eval budget in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin under 300s

        # Use a fixed seed for reproducibility within a run
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # For 5D, lambda = 4 + floor(3*ln(5)) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(dim))
        # mu = lam // 2
        mu = lam // 2
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff + 2)
        c_mu = min(1 - c_1,
                   (2 + (mu_eff / dim - 2) / (dim + 2)) / ((dim + 2) ** 2 + mu_eff + 5))
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_mu

        best_x = None
        best_f = np.inf

        # Multi-restart strategy: split budget into phases
        # Phase 1: broad exploration with larger sigma
        # Phase 2: refined search
        # We'll do up to 5 restarts, each with a portion of the budget

        # Total budget is 500 evaluations. Each CMA-ES iteration uses lam evaluations.
        # So max iterations per restart = budget_restarts / lam

        # Strategy: first restart gets more budget, subsequent ones less
        # Let's do: restart 1 gets 200 evals, restarts 2-5 get 60 each = 200+240=440, plus 60 spare
        # Actually let's be more aggressive: 5 restarts of ~100 evals each

        restart_budgets = []
        remaining = budget
        num_restarts = 5
        for i in range(num_restarts):
            if i == 0:
                rb = 200
            else:
                rb = (remaining - rb) // (num_restarts - i - 1) if i < num_restarts - 1 else remaining - sum(restart_budgets)
            rb = max(rb, lam * 2)  # at least 2 iterations
            restart_budgets.append(rb)
            remaining -= rb

        # Ensure total doesn't exceed budget
        total = sum(restart_budgets)
        if total > budget:
            # Scale down
            scale = budget / total
            restart_budgets = [int(rb * scale) for rb in restart_budgets]
            restart_budgets[-1] += budget - sum(restart_budgets)

        for restart_idx, rb in enumerate(restart_budgets):
            if time.time() > deadline - 5.0:
                break
            if rb < lam:
                break

            # Initialize CMA-ES
            if restart_idx == 0:
                # First restart: start at origin or random
                x_mean = np.zeros(dim)
                sigma = 2.0
            else:
                # Subsequent restarts: start near best found so far with perturbation
                if best_x is not None:
                    x_mean = best_x + 0.5 * rng.standard_normal(dim)
                    x_mean = np.clip(x_mean, -5, 5)
                else:
                    x_mean = rng.uniform(-4, 4, dim)
                sigma = 1.0

            # CMA-ES state
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            evals_used = 0

            max_iters = rb // lam

            for it in range(max_iters):
                if time.time() > deadline - 2.0:
                    break
                if evals_used >= rb:
                    break

                # Generate population
                Y = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    Y[i] = x_mean + sigma * (B @ (D * z))
                    Y[i] = np.clip(Y[i], -5, 5)

                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    F[i] = func(Y[i])
                    evals_used += 1
                    if time.time() > deadline - 1.0:
                        break

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Weighted mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += w[i] * Y[i]

                # Update step size (CSA)
                y_vec = (x_new - x_mean) / sigma
                p_sigma = (1 - c_1) * p_sigma + np.sqrt(c_1 * (2 - c_1) * mu_eff) * y_vec
                sigma *= np.exp((c_1 / d_amps) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_1) ** (2 * (it + 1) / (dim + c_1)))) \
                          / np.sqrt(mu_eff) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_1) ** (2 * (it + 1)))

                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu * h_sigma * delta_h) * C + \
                    c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(dim)) + \
                    c_mu * np.sum(w[i] * np.outer(Y[i] - x_mean, Y[i] - x_mean) / sigma ** 2 for i in range(mu))

                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    B, D = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
                    D = np.ones(dim)

                x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
