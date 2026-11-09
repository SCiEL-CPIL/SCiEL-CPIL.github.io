"""Rewritten optimizer using multi-start CMA-ES with adaptive budget allocation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with local refinement for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            x = x0.copy()
            sigma = sigma0
            fx = try_call(x)
            if fx is None:
                return best_x, best_f

            best_x_local = x.copy()
            best_f_local = fx

            # CMA-ES parameters
            mu = max(1, int(np.floor(3 * (dim + 1) / 2)))
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1) / ((dim + 1.3) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            inv_C = np.eye(dim)

            n_evals = 1
            stall_count = 0
            no_improve = 0

            while calls < max_calls and time.time() < deadline:
                # Generate lambda offspring
                lam = max(4, int(4 + 3 * dim))
                z = rng.standard_normal((lam, dim))
                # Transform using CMA-ES covariance
                # y = x + sigma * B @ (D * z)
                Y = np.array([x + sigma * (B @ (D * z[i])) for i in range(lam)])
                Y = np.clip(Y, -5.0, 5.0)

                F = np.full(lam, np.inf)
                for i in range(lam):
                    F[i] = try_call(Y[i])
                    if F[i] is None:
                        F[i] = np.inf
                    if F[i] < best_f_local:
                        best_f_local = F[i]
                        best_x_local = Y[i].copy()

                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]

                # Check for improvement
                if F[0] < fx - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1

                # Update mean
                m_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)

                # Update step size control variable
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * inv_C @ (m_new - x)))
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                    pass  # normal update
                else:
                    pass  # skip sigma adaptation when ps is large

                # Update covariance matrix
                v = (m_new - x) / sigma
                pc = (1 - c_c) * pc + c_c * v
                delta_hs = 1.0 if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma else 0.0

                # Rank-1 and rank-mu updates
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (delta_hs - 1 + c_c) * pc @ pc.T)
                C += c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(Y[:mu] - x, Y[:mu] - x) / (sigma ** 2), axis=0)

                # Ensure symmetry and positive definiteness
                C = 0.5 * (C + C.T)

                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

                x = m_new
                fx = F[0]
                n_evals += lam

                # Check for convergence
                if sigma < 1e-8 or no_improve > 30:
                    break

                # Stall detection
                if np.all(np.abs(Y[:mu] - x) < 1e-7):
                    break

            if fx < best_f_local:
                best_f_local = fx
                best_x_local = x.copy()

            return best_x_local, best_f_local

        def nelder_mead(x0, f0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            simplex = [x0.copy()]
            simplex_f = [f0]

            step = 0.15
            for i in range(dim):
                perturbation = x0.copy()
                perturbation[i] += step
                simplex.append(perturbation)

            for i in range(1, dim + 1):
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    break
                simplex_f.append(fx_i)
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            max_iterations = 80
            for _ in range(max_iterations):
                if calls >= max_calls or time.time() >= deadline:
                    break

                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                spread = max(simplex_f) - min(simplex_f)
                if spread < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = try_call(simplex[i])
                            if simplex_f[i] is None:
                                break
                        if simplex_f[-1] is None:
                            break

            return best_x, best_f

        def coordinate_descent(x0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            x = x0.copy()
            f = best_f
            step = 0.02
            no_improve = 0

            for _ in range(80):
                if calls >= max_calls or time.time() >= deadline:
                    break
                if no_improve > 25:
                    break

                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_call(x_try)
                        if f_try is None:
                            return best_x, best_f
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            no_improve = 0
                            break
                    if calls >= max_calls or time.time() >= deadline:
                        break

                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-9:
                        break
                else:
                    step = max(step * 1.05, 0.002)

            return best_x, best_f

        # Multi-start strategy
        # Start points: origin, and a few random points
        starts = [np.zeros(dim)]
        for _ in range(3):
            starts.append(rng.uniform(-3, 3, dim))

        # Allocate budget: first start gets 40%, subsequent get 20% each if budget allows
        # Then NM and CD on best result
        allocations = [0.40, 0.20, 0.20, 0.10]

        for i, start in enumerate(starts):
            if calls >= budget or time.time() >= deadline:
                break
            alloc = int(allocations[i] * budget) if i < len(allocations) else int(0.1 * budget)
            alloc = min(alloc, budget - calls)
            if alloc < 10:
                break

            sigma0 = 2.0 if i == 0 else 1.5
            best_x, best_f = cma_es(start, sigma0, alloc, deadline)

        # Local refinement with Nelder-Mead
        if best_x is not None and calls < budget and time.time() < deadline:
            nm_budget = min(60, budget - calls)
            if nm_budget > 10:
                best_x, best_f = nelder_mead(best_x, best_f, nm_budget, deadline)

        # Coordinate descent for final polish
        if best_x is not None and calls < budget and time.time() < deadline:
            cd_budget = budget - calls
            if cd_budget > 5:
                best_x, best_f = coordinate_descent(best_x, best_f, cd_budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
