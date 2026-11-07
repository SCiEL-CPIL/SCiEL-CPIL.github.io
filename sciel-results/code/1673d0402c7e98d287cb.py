"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_evals = 0

        def check_time():
            return time.time() < deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        def run_cma(x0, sigma_init, eval_budget, seed):
            nonlocal total_evals
            rng_local = np.random.default_rng(seed)
            
            x_mean = x0.copy()
            sigma = sigma_init
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            n_evals = 0
            no_improve = 0
            best_local = np.inf
            best_local_x = x0.copy()

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            c_sigma = (mueff + 0.5 + 1.0 / (dim + 1)) / (dim + 3 + 2.0 / (dim + 1))
            c_c = (4 + mueff / dim) / (dim + 4 + 2.0 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * ((mueff + 0.5) / (dim + 2) ** 2 - 1))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Evaluate initial
            fx = func(x_mean)
            n_evals += 1
            total_evals += 1
            update_best(x_mean, fx)
            best_local = fx
            best_local_x = x_mean.copy()

            while n_evals < eval_budget and check_time():
                # Generate population
                pop = []
                for i in range(lam):
                    if not check_time():
                        break
                    z = rng_local.standard_normal(dim)
                    y = np.clip(x_mean + sigma * z, -5, 5)
                    fy = func(y)
                    n_evals += 1
                    total_evals += 1
                    pop.append((fy, y))

                if not pop:
                    break

                pop.sort(key=lambda t: t[0])

                # Update best
                if pop[0][0] < best_local - 1e-12:
                    best_local = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                update_best(pop[0][1], pop[0][0])

                # Update mean
                x_old = x_mean.copy()
                x_new = np.zeros(dim)
                for i in range(min(mu, len(pop))):
                    x_new += w[i] * pop[i][1]
                x_mean = x_new

                # Update ps
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new - x_old) / sigma

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_old) / sigma

                # Update C
                C = (1 + c_1 - c_1 - c_mu) * C
                for i in range(min(mu, len(pop))):
                    diff = (pop[i][1] - x_old) / sigma
                    C += c_mu * w[i] * np.outer(diff, diff)
                C = 0.5 * (C + C.T)

                if no_improve > 40:
                    break

            return best_local_x, best_local, n_evals

        # Phase 1: CMA-ES with a few restarts
        num_restarts = 3
        evals_per = budget // num_restarts

        for r in range(num_restarts):
            if not check_time() or total_evals >= budget:
                break

            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.5

            remaining = min(evals_per, budget - total_evals)
            if remaining < lam + 1:
                remaining = budget - total_evals

            if remaining > 0:
                run_cma(x0, sigma0, remaining, 42 + r * 1000)

        # Phase 2: Local polish if time remains
        if best_x is not None and total_evals < budget and check_time():
            x = best_x.copy()
            fx = best_f
            step = 0.5
            remaining = budget - total_evals

            for _ in range(min(remaining, 30)):
                if not check_time():
                    break
                improved = False
                for i in range(dim):
                    for d in [1, -1]:
                        if not check_time():
                            break
                        x_try = x.copy()
                        x_try[i] += d * step
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = func(x_try)
                        total_evals += 1
                        if fx_try < fx - 1e-12:
                            x = x_try
                            fx = fx_try
                            improved = True
                            update_best(x, fx)
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
