import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            try:
                fx = func(x)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Phase 1: Scout with stratified sampling
        scout_budget = 30
        scout_points = []

        # Include center and some structured points
        points = [np.zeros(d)]
        # Add random points
        for _ in range(scout_budget - 1):
            points.append(rng.uniform(lo, hi, d))

        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = safe_call(x)
            scout_points.append((x, fx))

        if best_x is None:
            best_x = np.zeros(d)
            safe_call(best_x)

        # Phase 2: CMA-ES with better initialization and stagnation handling
        m = best_x.copy()
        
        # Initialize sigma based on scout spread
        if len(scout_points) > 2:
            scout_points.sort(key=lambda t: t[1])
            top_half = np.array([t[0] for t in scout_points[:max(1, len(scout_points) // 2)]])
            sigma = np.linalg.norm(np.std(top_half, axis=0), axis=0) * 0.5
            if sigma < 0.1:
                sigma = 0.1
            elif sigma > 2.0:
                sigma = 2.0
        else:
            sigma = 1.0

        lam = 4 + int(3 * np.log(d))
        if lam < 6:
            lam = 6

        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d * d))

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        remaining_budget = budget - call_count
        stagnation_count = 0
        stagnation_threshold = 20
        last_improvement_f = best_f

        while remaining_budget >= lam:
            if time.time() - self.start_time > self.deadline:
                break

            # Sample candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)

            Z = rng.standard_normal((lam, d))
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)

            F = np.array([safe_call(x) for x in X])
            remaining_budget -= lam

            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                last_improvement_f = best_f
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Stagnation handling
            if stagnation_count > stagnation_threshold:
                # Try a larger jump
                sigma *= 2.0
                if sigma > 5.0:
                    # Full restart from a new random point
                    new_x = rng.uniform(lo, hi, d)
                    new_f = safe_call(new_x)
                    remaining_budget -= 1
                    if new_f < best_f:
                        m = new_x.copy()
                        last_improvement_f = best_f
                    else:
                        # Keep m at current best
                        m = best_x.copy()
                    sigma = 1.0
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    stagnation_count = 0

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

            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)

            Y_delta = X_sorted[:mu] - m
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term

            C = (C + C.T) / 2.0
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)

            m = m_new

        # Phase 3: Local polish with smaller sigma
        if remaining_budget > 0 and time.time() - self.start_time < self.deadline:
            polish_budget = min(remaining_budget, 100)
            sigma_polish = max(sigma * 0.1, 0.01)
            m_polish = best_x.copy()
            C_polish = np.eye(d)
            pc_polish = np.zeros(d)
            ps_polish = np.zeros(d)

            for _ in range(polish_budget // lam):
                if time.time() - self.start_time > self.deadline:
                    break

                try:
                    L = np.linalg.cholesky(C_polish)
                except np.linalg.LinAlgError:
                    C_polish = np.eye(d)
                    L = np.eye(d)

                Z = rng.standard_normal((lam, d))
                Y = Z @ L.T
                X = m_polish[None, :] + sigma_polish * Y
                X = np.clip(X, lo, hi)

                F = np.array([safe_call(x) for x in X])

                sort_idx = np.argsort(F)
                X_sorted = X[sort_idx]

                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                eigvals, eigvecs = np.linalg.eigh(C_polish)
                eigvals = np.clip(eigvals, 1e-16, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                step = (m_new - m_polish) / sigma_polish
                ps_polish = (1 - c_sigma) * ps_polish + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)

                sigma_polish *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps_polish) / chi_n - 1))

                pc_polish = (1 - c_c) * pc_polish + np.sqrt(c_c * (2 - c_c) * mu_eff) * step

                h_sigma = 1 if np.linalg.norm(ps_polish) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                C_polish = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C_polish \
                    + c_1 * np.outer(pc_polish, pc_polish)

                Y_delta = X_sorted[:mu] - m_polish
                Y_delta = Y_delta / sigma_polish
                rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
                C_polish += c_mu * rank_mu_term

                C_polish = (C_polish + C_polish.T) / 2.0
                if not np.all(np.linalg.eigvalsh(C_polish) > 0):
                    C_polish = C_polish + 1e-4 * np.eye(d)

                m_polish = m_new

        return best_x, best_f
