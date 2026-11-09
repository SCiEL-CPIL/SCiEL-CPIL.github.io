import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 320.0  # 10s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # Phase 1: Scout Phase
        # Keep a modest scout to find a good starting point and estimate scale.
        scout_budget = 35
        best_x = None
        best_f = np.inf
        scout_points = []

        points = []
        points.append(np.zeros(d))
        # Add axis-aligned points to probe the structure
        for i in range(d):
            p = np.zeros(d)
            p[i] = 3.0
            points.append(p.copy())
            p[i] = -3.0
            points.append(p.copy())
        # Add random points
        for _ in range(scout_budget - (1 + 2*d)):
            points.append(rng.uniform(lo, hi, d))

        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        # Phase 2: CMA-ES
        lam = 4 + int(3 * np.log(d))
        if lam < 6:
            lam = 6

        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        chi_n = np.sqrt(d) * (1 - 1.0/(4*d) + 1.0/(21*d*d))

        def init_cma(start_x, start_sigma):
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            return m, sigma, C, pc, ps

        def cma_step(m, sigma, C, pc, ps, rng, func, lo, hi, lam, mu, weights,
                      mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)

            Z = rng.standard_normal((lam, d))
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)

            F = np.array([func(x) for x in X])

            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]

            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step

            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0

            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)

            Y_delta = X_sorted[:mu] - m
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term

            C = (C + C.T) / 2.0
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)

            return m_new, sigma, C, pc, ps, F_sorted[0], X_sorted[0]

        # Initialize from best scout point
        if len(scout_points) > 2:
            scout_points.sort(key=lambda t: t[1])
            top_quarter = np.array([t[0] for t in scout_points[:max(1, len(scout_points)//4)]])
            sigma_init = np.linalg.norm(np.std(top_quarter, axis=0), axis=0) * 0.7
            if sigma_init < 0.2:
                sigma_init = 0.2
            elif sigma_init > 1.5:
                sigma_init = 1.5
        else:
            sigma_init = 1.0

        m, sigma, C, pc, ps = init_cma(best_x, sigma_init)

        remaining_budget = budget - scout_budget
        best_f_cma = best_f
        best_x_cma = best_x.copy()

        stagnation_count = 0
        stagnation_threshold = 8

        while remaining_budget >= lam:
            if time.time() - self.start_time > self.deadline:
                break

            m, sigma, C, pc, ps, f_best_candidate, x_best_candidate = cma_step(
                m, sigma, C, pc, ps, rng, func, lo, hi, lam, mu, weights,
                mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, chi_n
            )

            if f_best_candidate < best_f_cma:
                best_f_cma = f_best_candidate
                best_x_cma = x_best_candidate.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            remaining_budget -= lam

            if stagnation_count >= stagnation_threshold:
                # Structural change: Instead of a complex restart or just increasing sigma,
                # we perform a "local re-seeding" using the best point found so far but with
                # a fresh, moderate sigma. This is simpler than the current program's mixed
                # restart strategy and avoids the overhead of evaluating 5 new points which
                # might not be in the right basin. We also reset the CMA-ES state to identity
                # to allow the covariance to adapt from scratch in the new neighborhood.
                # This mimics the "jolt" idea but is more controlled.
                restart_sigma = max(sigma * 2.0, 1.0)
                if restart_sigma > 3.0:
                    restart_sigma = 3.0
                
                # Re-initialize CMA-ES centered at the current best
                m, sigma, C, pc, ps = init_cma(best_x_cma, restart_sigma)
                stagnation_count = 0

        # Phase 3: Local Polish (Simulated Annealing)
        # Keep a small SA phase to fine-tune, but make it more aggressive in cooling
        # to quickly converge to the sharp minimum if close.
        if remaining_budget > 10 and time.time() - self.start_time < self.deadline:
            sa_budget = min(remaining_budget, 50)
            current_x = best_x_cma.copy()
            current_f = best_f_cma
            best_sa_x = best_x_cma.copy()
            best_sa_f = best_f_cma

            # Faster cooling and lower initial temp as per lessons
            T_initial = max(0.05, abs(best_f_cma) * 0.02 + 0.005)
            T_final = 1e-10
            cooling_rate = (T_final / T_initial) ** (1.0 / max(sa_budget - 1, 1))

            step_size = max(0.3, sigma * 0.5)
            
            for i in range(sa_budget):
                if time.time() - self.start_time > self.deadline:
                    break

                T = T_initial * (cooling_rate ** i)
                if T < T_final:
                    T = T_final

                neighbor_x = current_x + rng.standard_normal(d) * step_size
                neighbor_x = np.clip(neighbor_x, lo, hi)
                neighbor_f = func(neighbor_x)

                if neighbor_f < current_f:
                    current_x = neighbor_x.copy()
                    current_f = neighbor_f
                    if neighbor_f < best_sa_f:
                        best_sa_f = neighbor_f
                        best_sa_x = neighbor_x.copy()
                else:
                    delta_f = neighbor_f - current_f
                    if delta_f < 1000 * T:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            current_x = neighbor_x.copy()
                            current_f = neighbor_f
                            if neighbor_f < best_sa_f:
                                best_sa_f = neighbor_f
                                best_sa_x = neighbor_x.copy()

                # Faster decay
                step_size *= 0.95
                if step_size < 1e-5:
                    step_size = 1e-5

            if best_sa_f < best_f_cma:
                best_f_cma = best_sa_f
                best_x_cma = best_sa_x.copy()

        return best_x_cma, best_f_cma
