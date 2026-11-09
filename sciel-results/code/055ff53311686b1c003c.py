import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            fx = func(x)
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_cmaes(x0, sigma0, max_calls, rng_local, stagnation_limit=10):
            nonlocal calls_made
            lambda_ = 20
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 2) / (dim + mueff)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0
            local_calls = 0
            best_f_in_run = np.inf
            stagnation_count = 0
            prev_best = np.inf
            
            while local_calls + lambda_ <= max_calls and time.time() < deadline and calls_made < budget:
                B = rng_local.standard_normal((lambda_, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)
                
                Y = x_mean[None, :] + sigma * (B @ L)
                Y = np.clip(Y, -5, 5)

                F = []
                for y in Y:
                    if calls_made >= budget:
                        break
                    F.append(evaluate(y))
                    local_calls += 1
                
                if len(F) < lambda_:
                    return

                F = np.array(F)
                gen += 1
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]
                
                if F_sorted[0] < best_f_in_run:
                    best_f_in_run = F_sorted[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                if stagnation_count > stagnation_limit:
                    break

                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)
                
                step = (L_inv @ (x_mean - x_old).T).T[0]
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-16:
                    ps = (1 - c_sigma) * ps + c_sigma * step_norm / chi_n * (step / step_norm)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 50.0)

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (x_mean - x_old) / max(sigma, 1e-10)

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (Y_sorted[i] - x_old) / max(sigma, 1e-10)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (dim + 1 - c_1) - c_mu - c_c * (1 - c_c) * delta_hsig) * C \
                    + c_1 * (rank_1 - C) + c_mu * rank_mu

                C = (C + C.T) / 2
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-8:
                        C = C + 1e-8 * np.eye(dim)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

                if sigma < 1e-6 or gen > 50:
                    break

            return x_mean, best_f_in_run

        # Strategy
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        n_init = 10
        for _ in range(n_init):
            if calls_made >= budget: break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)
        
        # Phase 2: CMA-ES with Restarts
        if x_best is not None:
            x_start = x_best.copy()
            sigma_start = 1.0
            
            # Allocate most of the remaining budget to the first run
            remaining_budget = budget - calls_made
            first_run_budget = int(remaining_budget * 0.7)
            
            rng_local = np.random.default_rng(123)
            run_cmaes(x_start, sigma_start, first_run_budget, rng_local)
            
            # If there is still budget, do local restarts
            if calls_made < budget and x_best is not None:
                for _ in range(3):
                    if calls_made >= budget: break
                    x_restart = np.clip(x_best + 0.5 * rng.standard_normal(dim), -5, 5)
                    sigma_restart = 0.5
                    local_budget = min(50, budget - calls_made)
                    rng_local = np.random.default_rng(456)
                    run_cmaes(x_restart, sigma_restart, local_budget, rng_local)

        return x_best, f_best
