"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with CMA-ES-like adaptation and final local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initial point: try center and a few random points, pick best
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, dim))
        best_x = None
        best_f = np.inf
        for c in candidates:
            c = np.clip(c, -5, 5)
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        # CMA-ES-like parameters
        mu = 5  # number of parents
        lam = 15  # population size
        sigma = 1.5  # initial step size
        # Covariance matrix (identity to start)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Update weights
        w = np.log(lam / 2 + 1) - np.arange(lam)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        m = best_x.copy()
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 5  # initial candidates evaluated
        history = []  # (f, x) for all evaluations

        for f_val, x_val in [(best_f, best_x)]:
            history.append((f_val, x_val.copy()))

        # Main ES loop
        while calls < budget:
            # Check time
            if time.time() > deadline - 30:
                break

            # Generate population
            offspring = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                # Apply C^{-1/2} * z
                y = C_inv_sqrt @ z
                x_new = np.clip(m + sigma * y, -5, 5)
                offspring.append(x_new)

            # Evaluate
            offspring_f = []
            for x_new in offspring:
                if calls >= budget:
                    break
                f_new = func(x_new)
                calls += 1
                offspring_f.append(f_new)
                history.append((f_new, x_new.copy()))

            if len(offspring_f) == 0:
                break

            # Sort by fitness
            order = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in order]
            sorted_f = [offspring_f[i] for i in order]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Update mean (weighted average of top mu)
            m_new = m.copy()
            for i in range(min(mu, len(sorted_offspring))):
                m_new += w[i] * (sorted_offspring[i] - m)
            m = m_new

            # Update covariance
            v = np.zeros((dim, min(mu, len(sorted_offspring))))
            for i in range(min(mu, len(sorted_offspring))):
                v[:, i] = (sorted_offspring[i] - m) / sigma

            # Rank-mu update
            h_sigma = (
                np.sum(w[:min(mu, len(sorted_offspring))] ** 2) ** 0.5
                * (1 - np.exp(-c_sigma))
            )
            # Simplified: use rank-1 update
            v_mu = np.zeros(dim)
            for i in range(min(mu, len(sorted_offspring))):
                v_mu += w[i] * v[:, i]
            C = (1 - c_c * (2 - c_c)) * C + c_c * (
                np.outer(v_mu, v_mu) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim)
            )

            # Recompute C^{-1/2} (eigenvalue decomposition)
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + (
                c_sigma * d_sigma * chi_n / sigma
            ) * v_mu
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 50)

            # If stuck, restart with larger sigma
            if sigma < 0.05:
                sigma = 1.0
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        # Final local polish: Nelder-Mead style
        if time.time() < deadline - 5 and calls < budget:
            x = best_x.copy()
            f = best_f
            # Simple coordinate descent / local search
            step = 0.5
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x[d] + sign * step, -5, 5)
                        if calls >= budget:
                            break
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < f:
                            x, f = x_trial, f_trial
                            improved = True
                            history.append((f, x.copy()))
                    if calls >= budget:
                        break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
                else:
                    step = min(step * 1.2, 2.0)

            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
