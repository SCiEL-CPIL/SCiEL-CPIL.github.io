"""CMA-ES with Nelder-Mead polish for Bent Cigar (f12) optimization."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and Nelder-Mead polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 3 + 2 * mu_eff / n)
        c_cov_neg = (2 + mu_eff / n) / (n + 3 + 2 * mu_eff / n)

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        best_x = x_mean.copy()
        best_fx = func(x_mean)
        calls_used = 1

        # Initial covariance matrix square root
        D, B = np.linalg.eigh(C)
        D = np.maximum(D, 1e-12)
        C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

        # Stagnation tracking
        no_improve_count = 0
        stagnation_threshold = 12
        prev_best = best_fx

        for gen in range(1, 2000):
            if time.time() > deadline:
                break
            if calls_used + lam > budget:
                rem = budget - calls_used
                if rem < 1:
                    break
                for _ in range(rem):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                break

            # Sample population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * (C_inv_sqrt @ z)
                ys[i] = np.clip(ys[i], -5, 5)
                fs[i] = func(ys[i])
                calls_used += 1
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = ys[i].copy()

            # Check improvement
            if best_fx < prev_best - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            prev_best = best_fx

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]

            # Update mean
            x_mean_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

            # Update evolution paths
            y_vec = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

            # Update covariance
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.outer(pc, pc) + delta_h * np.outer(pc, pc)
            rank_1 = np.outer(y_vec, y_vec)
            C = (1 + c_sigma * (c_cov_pos - c_sigma) + c_cov_neg * (mu_eff - 2) - c_sigma * c_cov_neg) * C \
                + c_cov_pos * rank_1 + c_cov_neg * (np.sum(w) - 1) * rank_mu

            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update C^{-1/2}
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            x_mean = x_mean_new

            # Stagnation handling
            if no_improve_count >= stagnation_threshold and sigma > 0.05:
                x_mean = best_x.copy()
                sigma *= 2.0
                sigma = np.clip(sigma, 0.1, 50.0)
                C = np.eye(n)
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_count = 0

        # Nelder-Mead polish phase
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline - 10:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            step = max(sigma * 0.5, 0.1)
            for i in range(n):
                pt = best_x.copy()
                pt[i] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)
            simplex = np.array(simplex)
            fvals = np.array([best_fx] + [func(simplex[i]) for i in range(1, n + 1)])
            calls_used += n

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iters = min(remaining - 2, 100)

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid excluding worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                            if calls_used >= budget:
                                break

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

        return best_x, best_fx
