# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Hybrid multi-start optimizer for Katsuura function f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_evals, step=0.5):
            nonlocal best_f, best_x, evals_used
            n = dim
            # Create simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if time.time() > deadline or evals_used >= budget:
                    return
                fvals[i] = evaluate(simplex[i])

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if time.time() > deadline or evals_used >= budget:
                                return
                            fvals[i] = evaluate(simplex[i])
                        break

        def es_local(x0, max_evals, sigma0=0.3):
            nonlocal best_f, best_x, evals_used
            n = dim
            x = x0.copy()
            f = best_f
            sigma = sigma0
            C = np.eye(n)
            p_c = np.zeros(n)
            p_sigma = np.zeros(n)

            c_c = (4 + n) / (n + 4)
            c_sigma = 1 / (n + 3)
            c_1 = 2 / (n ** 2 + 18)
            c_mu = min(1 - c_1, 1 / (n + 2))
            chi_n = np.sqrt(n) * (1 - 1/(4*n))
            d_sigma = 1 + 2 * max(0, 1/np.sqrt(n+1) - 1) + c_sigma

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return
                if sigma < 1e-10:
                    break

                z = rng.standard_normal(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    C += 1e-6 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C_sqrt = np.eye(n)

                y = C_sqrt @ z
                x_trial = x + sigma * y
                f_trial = evaluate(x_trial)

                if f_trial < f:
                    y_sigma = y / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * y_sigma
                    h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2/(n+1)) * chi_n
                    if h_sigma:
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                    y_c = C @ y / sigma
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * y_c

                    delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                    C = (1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.outer(y/sigma, y/sigma)
                    x = x_trial
                    f = f_trial
                else:
                    # No update to mean, but still update paths with zero
                    pass

        # Phase 1: Structured diverse sampling
        # Center
        x_center = np.zeros(dim)
        evaluate(x_center)

        # Vertices of [-4, 4]^dim (since minimizer is in [-4,4]^5)
        vertices = []
        for i in range(2 ** dim):
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 8 - 4  # Map {0,1} to {-4,4}
            vertices.append(v)

        # Evaluate a subset of vertices (all 32 for dim=5)
        for v in vertices:
            if time.time() > deadline or evals_used >= budget:
                break
            evaluate(v)

        # Random points in [-4, 4]^dim
        n_random = 40
        for i in range(n_random):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 2: Nelder-Mead from best point
        remaining = budget - evals_used
        nm_budget = min(100, remaining // 3)
        nelder_mead(best_x.copy(), nm_budget, step=0.5)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 3: CMA-ES (1+1) local refinement
        remaining = budget - evals_used
        es_budget = min(200, remaining // 2)
        es_local(best_x.copy(), es_budget, sigma0=0.3)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 4: More Nelder-Mead from best
        remaining = budget - evals_used
        nm_budget2 = min(100, remaining // 3)
        nelder_mead(best_x.copy(), nm_budget2, step=0.3)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # Phase 5: Final random restarts + local refinement
        remaining = budget - evals_used
        n_restarts = 5
        per_restart = max(10, remaining // (n_restarts * 2))

        for r in range(n_restarts):
            if time.time() > deadline or evals_used >= budget:
                break

            # Sample near best point
            x_start = best_x + rng.normal(0, 1.0, dim)
            f_start = evaluate(x_start)

            if time.time() > deadline or evals_used >= budget:
                break

            # Quick Nelder-Mead
            nm_b = min(30, (budget - evals_used) // 2)
            if nm_b > 0:
                nelder_mead(x_start, nm_b, step=0.2)

        return best_x, best_f
# EVOLVE-BLOCK-END
