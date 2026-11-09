import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # 1. Scout Phase: Find a good starting point
        # Use ~10% of budget for scouting, min 20, max 30
        scout_budget = min(30, max(20, budget // 10))

        best_x = None
        best_f = np.inf
        scout_points = []

        # Start with center
        x0 = np.zeros(d)
        fx0 = func(x0)
        scout_points.append((x0, fx0))
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        for _ in range(scout_budget - 1):
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            scout_budget = 1

        # 2. CMA-ES Phase
        lam = 4 + int(3 * np.log(d))
        if lam < 8:
            lam = 8
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        m = best_x.copy()

        # Estimate initial sigma from scout points using mean of good points
        if len(scout_points) > 2:
            scout_xs = np.array([p[0] for p in scout_points])
            scout_fs = np.array([p[1] for p in scout_points])
            good_mask = scout_fs < np.percentile(scout_fs, 50)
            if np.sum(good_mask) >= 2:
                good_xs = scout_xs[good_mask]
                mean_good = np.mean(good_xs, axis=0)
                dists = np.linalg.norm(good_xs - mean_good, axis=1)
                sigma = max(0.5, np.mean(dists) * 0.5)
            else:
                sigma = 1.0
        else:
            sigma = 1.0

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        remaining_budget = budget - scout_budget
        best_f_cma = best_f
        best_x_cma = best_x.copy()

        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d * d))

        # Stagnation tracking for restarts
        no_improve_count = 0
        max_no_improve = lam * 5  # Restart if no improvement for 5 generations
        stagnation_threshold = 1e-8

        # Simulated Annealing parameters (used for restarts)
        T_initial = 2.0
        T_min = 1e-4
        cooling_rate = 0.95

        while remaining_budget >= lam:
            # Check time limit
            if time.time() - self.start_time > self.deadline:
                break

            # Check stagnation: if no improvement, trigger a restart with SA
            if no_improve_count >= max_no_improve:
                # Restart with a new point found via SA-like exploration
                # Use a few SA steps to find a new promising region
                T = max(T_initial * (cooling_rate ** no_improve_count), T_min)
                x_current = best_x_cma.copy()
                f_current = best_f_cma
                
                # Do a few SA steps to escape
                sa_steps = min(10, remaining_budget // 2)
                for _ in range(sa_steps):
                    if remaining_budget < 1:
                        break
                    x_new = x_current + sigma * rng.standard_normal(d)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = func(x_new)
                    remaining_budget -= 1
                    
                    if f_new < f_current:
                        x_current = x_new
                        f_current = f_new
                        if f_new < best_f_cma:
                            best_f_cma = f_new
                            best_x_cma = x_new.copy()
                    else:
                        # Accept with probability exp(-(f_new - f_current) / T)
                        if np.exp(-(f_new - f_current) / T) > rng.random():
                            x_current = x_new
                            f_current = f_new
                
                # Reset CMA-ES state with the new best point
                m = x_current.copy()
                sigma = max(0.5, sigma * 0.5)  # Reduce step size for local search
                C = np.eye(d)
                pc = np.zeros(d)
                ps = np.zeros(d)
                no_improve_count = 0
                T_initial = max(T_min, T_initial * 0.5)  # Reduce initial temperature for next restart
                continue

            # Sample candidates
            # Compute Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                C = np.eye(d)
                L = np.eye(d)

            # Sample standard normals
            Z = rng.standard_normal((lam, d))
            # Transform: x = m + sigma * (Z @ L.T)
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)

            # Evaluate
            F = np.array([func(x) for x in X])
            remaining_budget -= lam

            # Update best found globally
            min_idx = np.argmin(F)
            if F[min_idx] < best_f_cma:
                if best_f_cma - F[min_idx] > stagnation_threshold:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                best_f_cma = F[min_idx]
                best_x_cma = X[min_idx].copy()
            else:
                no_improve_count += 1

            # Sort indices by fitness
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]

            # Update mean m
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update evolution path ps
            # We need C^{-1/2}.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small eigenvalues
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step

            # Update covariance C
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0

            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)

            # Add rank-mu update
            Y_delta = X_sorted[:mu] - m  # Use old m for delta
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term

            # Enforce symmetry and positive definiteness (regularization)
            C = (C + C.T) / 2.0
            # Simple regularization if needed
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)

            m = m_new

        return best_x_cma, best_f_cma
