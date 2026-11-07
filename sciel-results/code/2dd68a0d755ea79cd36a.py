"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with coordinate-descent polish for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: CMA-ES with adaptive step size
        # Initialize at center with moderate spread
        x = np.zeros(dim)
        sigma = 3.0
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = 2 * dim  # population size
        mu = lambda_ // 2  # recombination weight count
        # weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        call_count = 1

        def time_ok():
            return time.time() < deadline and call_count < budget

        while time_ok():
            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # CMA-ES transformation
                Y[i] = x + sigma * (B @ (D * z))
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                call_count += 1
                if not time_ok():
                    break

            if call_count >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()

            # Mean update
            m_new = np.sum(weights[:, None] * Y[:mu], axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ ((m_new - x) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count // lambda_)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma

            # Update covariance
            delta_d = (hsig - 1) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - c_mu + delta_d) * C
            C += c_1 * (np.outer(pc, pc) - C)
            outer_sum = np.sum([np.outer(Y[i] - x, Y[i] - x) for i in range(mu)], axis=0)
            C += c_mu / sigma ** 2 * outer_sum

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Eigen decomposition for sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)
                C_inv = (eigvecs @ np.diag(1.0 / D) @ eigvecs.T)
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = m_new
            x = np.clip(x, -5, 5)

        # Phase 2: Coordinate descent polish
        if time_ok():
            x = best_x.copy()
            fx = best_fx
            step = 0.5
            for _ in range(30):
                improved = False
                for coord in range(dim):
                    for direction in [-1, 1]:
                        trial = x.copy()
                        trial[coord] += direction * step
                        if -5 <= trial[coord] <= 5:
                            ft = func(trial)
                            call_count += 1
                            if ft < fx:
                                x = trial
                                fx = ft
                                improved = True
                                if ft < best_fx:
                                    best_fx = ft
                                    best_x = x.copy()
                            if not time_ok():
                                break
                    if not time_ok():
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                if not time_ok():
                    break

        # Phase 3: Small random perturbations around best
        if time_ok():
            pert_sigma = 0.1
            for _ in range(20):
                trial = best_x + pert_sigma * rng.standard_normal(dim)
                trial = np.clip(trial, -5, 5)
                ft = func(trial)
                call_count += 1
                if ft < best_fx:
                    best_fx = ft
                    best_x = trial.copy()
                if not time_ok():
                    break
                pert_sigma *= 0.8

        return best_x, best_fx
# EVOLVE-BLOCK-END
