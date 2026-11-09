"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal Rank-1 + Rank-mu Natural Gradient Descent with stagnation-triggered restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_params(self, dim):
        lambda_ = 4 * dim
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        return dict(lambda_=lambda_, mu=mu, w=w, mu_eff=mu_eff, c_sigma=c_sigma,
                    c_c=c_c, c_1=c_1, c_mu=c_mu, d_sigma=d_sigma, chi_n=chi_n)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        P = self._cma_params(dim)
        lambda_ = P["lambda_"]
        mu = P["mu"]
        w = P["w"]
        mu_eff = P["mu_eff"]
        c_sigma, c_c, c_1, c_mu, d_sigma, chi_n = (
            P["c_sigma"], P["c_c"], P["c_1"], P["c_mu"], P["d_sigma"], P["chi_n"])

        best_x = None
        best_fx = np.inf
        calls = 0

        def eval_point(y):
            nonlocal best_x, best_fx, calls
            fy = func(y)
            calls += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            return fy

        # Initial diverse sampling: capture multiple basins early for AOCC
        for _ in range(min(10, self.budget)):
            y = rng.uniform(-5, 5, dim)
            eval_point(y)
        if calls >= self.budget:
            return best_x, best_fx

        # Stagnation detection
        window = 3 * lambda_
        best_at_window_start = best_fx
        calls_at_window_start = calls
        restarts = 0

        while calls < self.budget and time.time() < deadline:
            # Initialize / re-initialize state
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
                eval_point(x)
            else:
                x = best_x.copy()
            sigma = 2.0
            # Diagonal step sizes (independent per dimension)
            sigma_vec = np.ones(dim) * sigma
            # Evolution path for sigma adaptation
            ps = np.zeros(dim)
            # Evolution path for diagonal covariance (rank-1 update)
            pc = np.zeros(dim)
            # Diagonal of covariance (log scale for stability)
            log_diag = np.zeros(dim)  # log(sigma_i^2)

            # Run diagonal CMA-ES until stagnation or budget/time
            while calls < self.budget and time.time() < deadline:
                remaining = self.budget - calls
                lam_cur = min(lambda_, remaining)
                if lam_cur < 2:
                    break

                ys = []
                fs = []
                for i in range(lam_cur):
                    z = rng.standard_normal(dim)
                    # Sample using diagonal step sizes
                    y = x + sigma_vec * z
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    ys.append(y)
                    fs.append(fy)
                    if time.time() > deadline:
                        break

                if len(ys) < mu:
                    break

                idx = np.argsort(fs)[:mu]
                ys_sorted = [ys[i] for i in idx]

                x_old = x.copy()
                x = sum(w[i] * ys_sorted[i] for i in range(mu))

                # Update evolution path ps (for sigma adaptation)
                ps = (1 - c_sigma) * ps + \
                    np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * ((x - x_old) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** ((calls + 1) // lambda_))) < \
                    (1.4 + 2 / (dim + 1)) * chi_n

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update evolution path pc (for diagonal covariance)
                pc = (1 - c_c) * pc + hsig * \
                    np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)

                # Update diagonal covariance via rank-1 + rank-mu
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                # Rank-1 update
                diag_update = c_1 * delta_hsig * np.outer(pc, pc).diagonal()
                # Rank-mu update
                rank_mu = c_mu / sigma ** 2 * np.array(
                    [sum(w[i] * (ys_sorted[i][d] - x_old[d]) ** 2 for i in range(mu)) for d in range(dim)])
                
                # Combine updates on log scale for stability
                log_diag = log_diag + (c_1 * delta_hsig - c_1 - c_mu) * log_diag + \
                    np.log(np.maximum(diag_update + rank_mu + np.exp(log_diag) * (1 - c_1 * delta_hsig + c_1 + c_mu), 1e-12))
                
                # Actually, let's simplify: direct update on sigma_vec
                sigma_vec = np.sqrt(np.maximum(np.exp(log_diag), 1e-12))

                # Clip sigma_vec to reasonable range
                sigma_vec = np.clip(sigma_vec, 1e-10, 10.0)

            # Stagnation check: population best not improving over the window
            improved = best_at_window_start - best_fx
            if improved < 1e-6 and calls - calls_at_window_start >= window and restarts < 4:
                # Diverse restart: fresh mean near the global best, identity covariance, larger sigma
                x = best_x.copy() + rng.normal(0, 1.5, dim)
                x = np.clip(x, -5, 5)
                restarts += 1
            else:
                # Reset the window regardless
                best_at_window_start = best_fx
                calls_at_window_start = calls

        return best_x, best_fx
# EVOLVE-BLOCK-END
