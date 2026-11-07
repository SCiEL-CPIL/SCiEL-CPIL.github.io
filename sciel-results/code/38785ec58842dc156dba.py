import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        global_best_x = None
        global_best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(12345)

        # Phase 1: random exploration to get an early good estimate (helps AOCC early)
        for _ in range(15):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = float(func(x))
            eval_count += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        # Phase 2: CMA-ES with restarts on no-improvement
        max_restarts = 15
        for restart in range(max_restarts):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            stop, x_best, f_best, evals_used = self._cma_es(
                func, dim, global_best_x, global_best_f,
                eval_count, budget, deadline, rng)
            eval_count += evals_used
            if f_best < global_best_f:
                global_best_f = f_best
                global_best_x = x_best.copy()
            # If CMA-ES stopped without finding better, restart from a fresh random point
            if stop == 'no_improvement':
                # Try a few random restarts to escape
                found = False
                for _ in range(3):
                    if eval_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = float(func(x))
                    eval_count += 1
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                        found = True
                        break
                if not found:
                    # nudge from best
                    x = np.clip(global_best_x + 2.0 * rng.standard_normal(dim), -5, 5)
                    fx = float(func(x))
                    eval_count += 1
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
            if stop == 'budget' or stop == 'time':
                break

        return global_best_x, global_best_f

    def _cma_es(self, func, dim, x0, f0, eval_count, budget, deadline, rng):
        """Run one CMA-ES instance starting near x0. Returns (stop_reason, x_best, f_best, evals_used)."""
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        c_sigma = (4 + lam / (dim + 4)) / ((dim + 1.3) ** 2 + lam)
        c_c = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_cov = (mu_eff + 2) / (dim ** 1.7 + mu_eff)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        sigma = 2.0  # initial step size

        x_mean = x0.copy().astype(float)
        f_best = f0
        x_best = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        evals_used = 0
        best_in_run = f0
        no_improve_evals = 0
        patience = 3 * lam  # evaluations without improvement before declaring no-improvement

        while eval_count + evals_used < budget:
            if time.time() > deadline:
                return 'time', x_best, f_best, evals_used

            # Sample population
            B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
            # Standardize
            norms = np.linalg.norm(B, axis=1)
            norms[norms < 1e-12] = 1e-12
            B = B / norms[:, None]
            # Evaluate
            Y = x_mean[None, :] + sigma * B
            Y = np.clip(Y, -5, 5)
            F = np.empty(lam)
            for i in range(lam):
                F[i] = float(func(Y[i]))
                evals_used += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = Y[i].copy()
                    if F[i] < best_in_run:
                        best_in_run = F[i]
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                else:
                    no_improve_evals += 1
                if eval_count + evals_used >= budget:
                    break

            if eval_count + evals_used >= budget:
                return 'budget', x_best, f_best, evals_used

            if no_improve_evals >= patience:
                return 'no_improvement', x_best, f_best, evals_used

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            Yw = Y[:mu]

            # Update mean
            x_new_mean = x_mean + np.sum(weights[:, None] * (Yw - x_mean[None, :]), axis=0)
            x_new_mean = np.clip(x_new_mean, -5, 5)

            # Update step-size control
            B_inv = np.linalg.cholesky(C)
            try:
                v = np.linalg.solve(B_inv, (Yw - x_mean[None, :]).T).T
            except np.linalg.LinAlgError:
                v = (Yw - x_mean[None, :])
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv.T @ (weights @ v))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (evals_used + 1)) / chi_n) < (1.4 + 2 / (dim + 1))

            # Update covariance
            delta_h = (1.0 if hsig else 0.0) * (1 - c_c)
            C = (1 - c_cov * (c_cov + 2 * c_c)) * C + \
                c_cov * (delta_h * pc @ pc.T + (weights[:, None] * (Yw - x_mean[None, :])) @ (weights[:, None] * (Yw - x_mean[None, :])).T)
            # Symmetrize and ensure PSD
            C = 0.5 * (C + C.T)
            # Regularize
            C += 1e-12 * np.eye(dim)

            # Update cumulation
            pc = (1 - c_c) * pc + delta_h * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update sigma
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            x_mean = x_new_mean

        return 'budget', x_best, f_best, evals_used
