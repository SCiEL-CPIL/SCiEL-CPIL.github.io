# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer for BBOB f7 (Step ellipsoidal):
    Phase 1: Multi-start L-BFGS-B via scipy (exploits smoothness within basin)
    Phase 2: CMA-ES with restarts for robust global refinement
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = np.array(x, dtype=float, copy=True)
            return val

        def time_ok():
            return time.time() < deadline and self._call_count < budget

        # --- Phase 1: Multi-start L-BFGS-B (30% of budget) ---
        from scipy.optimize import minimize

        n_starts = 10
        starts = [np.zeros(dim)]
        # Add structured starting points
        for i in range(1, n_starts):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Shuffle
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        for x0 in starts:
            if not time_ok():
                break
            remaining_for_lbfgs = int(budget * 0.3) - self._call_count
            if remaining_for_lbfgs <= 5:
                break
            try:
                res = minimize(count_eval, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8,
                                        'maxls': 20})
            except Exception:
                continue

        # --- Phase 2: CMA-ES for global refinement (remaining budget) ---
        if not time_ok():
            return self._best_x, self._best_f

        if self._best_x is None:
            x_mean = rng.uniform(-5, 5, dim)
        else:
            x_mean = self._best_x.copy()

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        mu = lam // 2
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
        sigma = 0.5

        stall_count = 0
        prev_best_f = self._best_f
        stall_threshold = 4 * lam
        no_improve_counter = 0
        restart_count = 0
        max_restarts = 8

        while time_ok() and self._call_count + lam <= budget:
            # Generate population
            population = []
            fvals = []
            
            # Try to use Cholesky once per generation for efficiency
            try:
                L = np.linalg.cholesky(C)
                use_cholesky = True
            except np.linalg.LinAlgError:
                L = None
                use_cholesky = False

            for i in range(lam):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                if use_cholesky:
                    x_trial = x_mean + sigma * (L @ z)
                else:
                    x_trial = x_mean + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                f = count_eval(x_trial)
                population.append(x_trial)
                fvals.append(f)

            if len(fvals) < lam:
                break

            # Sort by fitness
            sorted_indices = np.argsort(fvals)
            selected = [population[i] for i in sorted_indices[:mu]]
            selected_f = [fvals[i] for i in sorted_indices[:mu]]

            # Check improvement
            if selected_f[0] < prev_best_f - 1e-10:
                prev_best_f = selected_f[0]
                stall_count = 0
                no_improve_counter = 0
            else:
                stall_count += lam
                no_improve_counter += lam

            # Compute weighted mean
            x_new = np.zeros(dim)
            for w, x_sel in zip(weights, selected):
                x_new += w * x_sel

            # Update sigma
            diff = x_new - x_mean
            if np.linalg.norm(diff) > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * chi_n)
                ps_norm = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)
                if ps_norm > 1.4 * chi_n + 1:
                    sigma *= 0.8
                elif ps_norm < (1.4 - 1 / c_sigma) * chi_n:
                    sigma *= 1.2
                    sigma = min(sigma, 10.0)

            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(
                1 - (1 - c_sigma) ** 2) < (1.4 + 1 / c_sigma) * chi_n else 0

            rank_one = np.outer(p_c, p_c) - C
            rank_mu = np.zeros((dim, dim))
            for w, x_sel in zip(weights, selected):
                diff_sel = (x_sel - x_mean) / sigma
                rank_mu += w * np.outer(diff_sel, diff_sel)

            C = (1 + c_1 * h_sigma * mu_eff - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2

            # Regularize
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] < 1e-10:
                C += 1e-10 * np.eye(dim)

            x_mean = x_new

            # Restart logic
            if stall_count > stall_threshold or no_improve_counter > 6 * lam:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart from best found with perturbation
                if self._best_x is not None:
                    x_mean = self._best_x + rng.normal(0, 0.3, dim)
                    x_mean = np.clip(x_mean, -5, 5)
                else:
                    x_mean = rng.uniform(-5, 5, dim)
                sigma = max(sigma, 0.5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                no_improve_counter = 0

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
