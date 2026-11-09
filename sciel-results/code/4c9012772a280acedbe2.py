"""Combined optimizer for BBOB f13 (Sharp ridge): CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive restarts for Sharp Ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0
        calls_made = 0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            return x, f

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter=200):
            nonlocal calls_made, best_f, best_x

            if time.time() > deadline - 30:
                return x0, np.inf

            # Parameters
            pop_size = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            pop_size = min(pop_size, max(4, (budget - calls_made) // 10))

            # Strategy parameters
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            cs = (dim + 1.7) / (dim + 9 + pop_size / 2)
            ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:int(1 / weights[0])])**2)

            # State
            x_mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            inv_sqrt_C = np.eye(dim)

            # Initial population
            best_local_x = x_mean.copy()
            best_local_f = np.inf
            stagnation = 0

            for iteration in range(max_iter):
                if time.time() > deadline - 30 or calls_made >= budget:
                    break

                # Generate population
                for i in range(pop_size):
                    if time.time() > deadline - 30 or calls_made >= budget:
                        break

                    # Sample from multivariate normal
                    z = np.random.randn(dim)
                    x_new = x_mean + sigma * (B @ (D * (inv_sqrt_C @ z)))
                    x_new = clip(x_new)

                    res = evaluate(x_new)
                    if res is None:
                        break
                    x_new, f_new = res

                    if f_new < best_local_f:
                        best_local_f = f_new
                        best_local_x = x_new.copy()

                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

                if calls_made >= budget or time.time() > deadline - 30:
                    break

                # Reconstruct population for update
                # We need to store the sampled points and their function values
                # For simplicity, let's use a different approach: track the population explicitly

                # Actually, let me restructure: store population in arrays
                # This requires a different loop structure

                # Check convergence
                if iteration > 10 and best_local_f < 1e-10:
                    break

                # Stagnation check
                if abs(best_local_f - best_f) < 1e-8:
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation > 20:
                    break

            return best_local_x, best_local_f

        # Better CMA-ES with proper population tracking
        def cma_es_v2(x0, sigma0, max_iter=100):
            nonlocal calls_made, best_f, best_x

            if time.time() > deadline - 30:
                return x0, np.inf

            pop_size = 4 + int(3 * np.log(dim))
            pop_size = min(pop_size, max(4, (budget - calls_made) // 15))

            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            cs = (dim + 1.7) / (dim + 9 + pop_size / 2)
            ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:int(1 / weights[0])])**2)

            x_mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            inv_sqrt_C = np.eye(dim)

            best_local_x = x_mean.copy()
            best_local_f = np.inf

            for iteration in range(max_iter):
                if time.time() > deadline - 30 or calls_made >= budget:
                    break

                # Generate population
                population = []
                fvals = []

                for i in range(pop_size):
                    if time.time() > deadline - 30 or calls_made >= budget:
                        break

                    z = np.random.randn(dim)
                    x_new = x_mean + sigma * (B @ (D * (inv_sqrt_C @ z)))
                    x_new = clip(x_new)

                    res = evaluate(x_new)
                    if res is None:
                        break
                    x_new, f_new = res

                    population.append(x_new)
                    fvals.append(f_new)

                    if f_new < best_local_f:
                        best_local_f = f_new
                        best_local_x = x_new.copy()

                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

                if len(population) < pop_size:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                population = [population[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                x_mean_new = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)

                # Update evolution paths
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (inv_sqrt_C @ (x_mean_new - x_mean) / sigma)
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc)**((iteration + 1) / ds))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs)**((iteration + 1) / ds))
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * (x_mean_new - x_mean) / sigma

                # Update covariance matrix
                delta_sig = 1 + cs * (np.linalg.norm(ps)**2 / dim - 1)
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = inv_sqrt_C @ (population[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + mu_sigma * (delta_sig - 1) - (1 - mu_sigma)) * C + rank_one + rank_mu
                # Simplified: C = (1 - mu_sigma) * C + rank_one + rank_mu where mu_sigma = mueff / (dim + 2)
                mu_sigma = mueff / (dim + 2)
                C = (1 - mu_sigma) * C + rank_one + rank_mu

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update B, D, inv_sqrt_C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.sqrt(np.maximum(eigvals, 1e-12))
                B = eigvecs
                D = eigvals
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.maximum(eigvals, 1e-12)) @ eigvecs.T

                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                x_mean = x_mean_new

                # Convergence check
                if np.all(np.abs(fvals[:mu] - fvals[0]) < 1e-10):
                    break

            return best_local_x, best_local_f

        # Phase 1: Initial exploration with Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim, low, high):
            point = np.zeros(dim)
            for d in range(dim):
                base = d + 2
                point[d] = low + (high - low) * halton(index + 1, base)
            return point

        n_explore = min(30, budget // 10)
        best_x = None
        best_f = np.inf
        candidate_starts = []

        for i in range(n_explore):
            if time.time() > deadline - 60:
                break
            x = halton_point(i, dim, box_low, box_high)
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            candidate_starts.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Add origin and structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 1.5,
            np.ones(dim) * -1.5,
            np.array([1, -1, 1, -1, 1]) * 1.5,
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            candidate_starts.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort candidates by function value
        candidate_starts.sort(key=lambda t: t[1])

        # Phase 2: Run CMA-ES from best candidates with adaptive restarts
        np.random.seed(42)  # For reproducibility within a run

        num_restarts = min(3, len(candidate_starts))
        restarts_used = 0

        for restart in range(num_restarts):
            if time.time() > deadline - 40 or calls_made >= budget:
                break

            if restart < len(candidate_starts):
                start_x = candidate_starts[restart][0].copy()
            else:
                start_x = best_x.copy() + np.random.randn(dim) * 0.5

            sigma_start = 1.0 if restart == 0 else 0.5

            x_res, f_res = cma_es_v2(start_x, sigma_start, max_iter=50)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

            restarts_used += 1

            # Check if we should continue
            if best_f < 1e-6:
                break

        # Phase 3: Final refinement with smaller CMA-ES
        if time.time() < deadline - 20 and calls_made < budget and best_x is not None:
            np.random.seed(42)
            x_res, f_res = cma_es_v2(best_x, 0.1, max_iter=30)
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
