"""Model-and-optimize: CMA-ES with radial sweep initialization and step-size refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with radial sweep for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES starting from x0 with initial step size sigma0."""
            nonlocal call_count, best_f, best_x

            # Initialize population
            n = dim
            lambda_ = 10  # Larger population for better exploration
            mu = lambda_ // 2
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_c = min(c_c, 1.0)
            mueff_inv = 1.0 / mueff
            c_1 = (2 - mueff_inv + c_sigma) / ((n + 1.3) ** 2 + mueff_inv + c_sigma)
            c_mu = (2 * (mueff_inv - 1) + c_sigma) / ((n + 2) ** 2 + mueff_inv + c_sigma)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff_inv - 1) / (n + 1)) - 1) + c_sigma

            # Covariance matrix adaptation
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            evals = 0

            # Initial population
            pop_x = []
            pop_f = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (B @ (D * z))
                fi = try_eval(xi)
                if fi is None:
                    break
                pop_x.append(xi)
                pop_f.append(fi)
                evals += 1
                if evals >= max_evals:
                    break

            if len(pop_x) < 2:
                return

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]

            while evals < max_evals and call_count < budget:
                if time.time() > deadline - 10:
                    break

                # Update weights
                w = weights[:min(mu, len(pop_f))]
                x_mean_new = np.sum(np.array(pop_x[:len(w)]) * w[:, None], axis=0)

                # Update translation (cm)
                pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
                hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (evals + 1))) / chi_n < 1.4 + 2 / (n + 1)
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hs) * c_c * (2 - c_c) * C) + \
                    c_mu * np.sum(w[:, None, None] * (np.array(pop_x[:len(w)]) - x_mean) @ (np.array(pop_x[:len(w)]) - x_mean).T, axis=0)

                # Eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B * np.sqrt(D[:, None])

                # Update sigma (cs)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e6)
                sigma = max(sigma, 1e-12)

                x_mean = x_mean_new

                # Generate new population
                pop_x_new = []
                pop_f_new = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (B @ z)
                    fi = try_eval(xi)
                    if fi is None:
                        break
                    pop_x_new.append(xi)
                    pop_f_new.append(fi)
                    evals += 1
                    if evals >= max_evals:
                        break

                if len(pop_x_new) < 2:
                    break

                idx_new = np.argsort(pop_f_new)
                pop_x = [pop_x_new[i] for i in idx_new]
                pop_f = [pop_f_new[i] for i in idx_new]

        # Phase 1: Light initialization
        # Evaluate origin
        try_eval(np.zeros(dim))
        
        # Evaluate points along axes to help identify the narrow dimension
        # Radii: 0.5, 1.0, 2.0, 4.0
        radii = [0.5, 1.0, 2.0, 4.0]
        for i in range(dim):
            for sign in [-1, 1]:
                for r in radii:
                    if call_count >= budget:
                        break
                    x = np.zeros(dim)
                    x[i] = sign * r
                    try_eval(x)
                if call_count >= budget:
                    break
            if call_count >= budget:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: CMA-ES
        if call_count < budget:
            x0 = best_x.copy()
            # Larger initial sigma to allow faster adaptation to the anisotropic landscape
            sigma0 = 1.5
            cma_es(x0, sigma0, budget - call_count)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
