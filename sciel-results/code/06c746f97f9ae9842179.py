"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: L-BFGS-B for rapid convergence + CMA-ES for robust refinement on BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_f = np.inf

        def count_eval(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: L-BFGS-B from multiple diverse starting points for rapid initial convergence
        # The Step Ellipsoidal function has a relatively smooth basin around the minimum,
        # so gradient-based methods can make fast progress once close enough.
        n_starts = 10
        starts = [np.zeros(self.dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))

        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        for x0 in starts:
            if call_count >= self.budget * 0.35:  # Use 35% of budget for L-BFGS-B
                break
            if time.time() > deadline:
                break

            try:
                res = minimize(count_eval, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-7})
            except Exception:
                continue

        # Phase 2: CMA-ES for robust refinement that handles step ellipsoidal geometry
        if call_count < self.budget - 15 and time.time() < deadline:
            if best_x is None:
                x_mean = rng.uniform(-5, 5, self.dim)
            else:
                x_mean = best_x.copy()

            dim = self.dim
            # CMA-ES parameters tuned for dim=5, budget ~325 remaining
            lam = 4 + int(3 * np.log(dim))  # population size ~14
            mu = lam // 2  # number of parents
            weights = np.ones(mu) / mu
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            C = np.eye(dim)
            sigma = 0.3  # Start with smaller step since L-BFGS-B should be close

            stall_count = 0
            prev_best_f = best_f
            stall_threshold = 4 * lam

            while call_count + lam <= self.budget and time.time() < deadline:
                # Generate population
                population = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    try:
                        L = np.linalg.cholesky(C)
                        x_trial = x_mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        x_trial = x_mean + sigma * z
                    x_trial = np.clip(x_trial, -5, 5)
                    population.append(x_trial)

                # Evaluate population
                fvals = []
                for x_trial in population:
                    if call_count >= self.budget or time.time() >= deadline:
                        break
                    f = count_eval(x_trial)
                    fvals.append(f)

                if len(fvals) < lam:
                    break

                # Sort by fitness
                sorted_indices = np.argsort(fvals)
                selected = [population[i] for i in sorted_indices[:mu]]
                selected_f = [fvals[i] for i in sorted_indices[:mu]]

                # Check for improvement
                if selected_f[0] < prev_best_f - 1e-10:
                    prev_best_f = selected_f[0]
                    stall_count = 0
                else:
                    stall_count += lam

                # Compute weighted mean of selected
                x_new = np.zeros(dim)
                for w, x_sel in zip(weights, selected):
                    x_new += w * x_sel

                # Update sigma control
                diff = x_new - x_mean
                if np.linalg.norm(diff) > 1e-16:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * chi_n)
                    norm_ps = np.linalg.norm(p_sigma)
                    if norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) > 1.4 * chi_n + 1:
                        sigma *= 0.8
                    elif norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 - 1 / c_sigma) * chi_n:
                        sigma *= 1.2
                        if sigma > 5:
                            sigma = 5

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / c_sigma) * chi_n else 0

                rank_one = np.outer(p_c, p_c) - C
                rank_mu = np.zeros((dim, dim))
                for w, x_sel in zip(weights, selected):
                    diff_sel = (x_sel - x_mean) / sigma
                    rank_mu += w * np.outer(diff_sel, diff_sel)

                C = (1 + c_1 * h_sigma * mu_eff - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2  # Ensure symmetry

                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if eigvals[0] < 1e-10:
                    C += 1e-10 * np.eye(dim)

                x_mean = x_new

                # Restart if stalled
                if stall_count > stall_threshold:
                    if best_x is not None:
                        x_mean = best_x + rng.uniform(-0.3, 0.3, dim)
                        x_mean = np.clip(x_mean, -5, 5)
                    sigma = max(sigma, 0.2)
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    stall_count = 0

        # Phase 3: Final local refinement with (1+1)-ES if budget remains
        if best_x is not None and call_count < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while call_count < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = count_eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                if sigma < 1e-7:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
