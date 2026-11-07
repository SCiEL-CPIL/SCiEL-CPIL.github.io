"""Multi-start optimizer: Sobol global + CMA-ES adaptive refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import qmc


class Optimizer:
    """Multi-start optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with Sobol sequence
        n_global = 30
        if n_global > budget:
            n_global = budget

        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = 1
        while m < n_global:
            m *= 2
        sobol_points = sobol.random(m)
        sobol_points = sobol_points * 10.0 - 5.0

        # Track all evaluated points for multi-start
        evaluated_points = []
        evaluated_fvals = []

        for i in range(n_global):
            if calls_made >= budget or time_left() < 5:
                break
            x = sobol_points[i]
            f = make_call(x)
            if f is not None:
                evaluated_points.append(x.copy())
                evaluated_fvals.append(f)

        if best_x is None:
            make_call(np.zeros(dim))
            return best_x, best_f

        # Sort evaluated points by fitness (best first)
        if len(evaluated_points) > 0:
            sort_idx = np.argsort(evaluated_fvals)
            evaluated_points = [evaluated_points[i] for i in sort_idx]
            evaluated_fvals = [evaluated_fvals[i] for i in sort_idx]

        # Phase 2: CMA-ES refinement on top candidates
        # CMA-ES handles rotated landscapes better than Nelder-Mead

        def cma_es(start_x, remaining_budget, sigma_init=0.5):
            nonlocal calls_made, best_x, best_f

            if calls_made >= budget or time_left() < 2:
                return

            # CMA-ES parameters
            lam = 15  # population size
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)

            # Learning rates
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

            # Initialize CMA-ES
            m = start_x.copy()
            sigma = sigma_init

            # State variables
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            dcor = 1.0 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

            # Eigen decomposition
            C_eigvals, C_eigvecs = np.linalg.eigh(C)

            def update_eig():
                nonlocal C, C_eigvals, C_eigvecs
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-12)

            rng_local = np.random.default_rng(123)

            # CMA-ES loop
            while calls_made < budget and time_left() > 2:
                # Compute square root of C
                sqrt_C_eigvals = np.sqrt(C_eigvals)
                C_sqrt = C_eigvecs @ np.diag(sqrt_C_eigvals)

                # Generate population
                Z = rng_local.standard_normal((lam, dim))
                Y = m[None, :] + sigma * (C_sqrt @ Z.T).T
                Y = np.clip(Y, -5.0, 5.0)

                # Evaluate population
                F = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_made >= budget or time_left() < 1:
                        break
                    f = make_call(Y[i])
                    if f is not None:
                        F[i] = f

                if calls_made >= budget or time_left() < 1:
                    break

                # Check if any evaluation was made
                if np.all(np.isinf(F)):
                    break

                # Sort by fitness
                valid = ~np.isinf(F)
                if not np.any(valid):
                    break
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]

                # Update mean
                m_old = m.copy()
                valid_idx = [i for i in idx if not np.isinf(F[i])]
                if len(valid_idx) < mu:
                    sel_Y = Y[valid_idx]
                    sel_w = np.ones(len(valid_idx)) / len(valid_idx)
                    m = np.sum(sel_w[:, None] * sel_Y, axis=0)
                else:
                    Y_sorted_valid = Y_sorted[:mu]
                    m = np.sum(w[:, None] * Y_sorted_valid, axis=0)

                # Update sigma, C, pc, ps
                if sigma > 1e-12:
                    y = (m - m_old) / sigma
                else:
                    y = np.zeros(dim)

                # Update ps
                hs = (math.sqrt(1 - math.exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
                if np.linalg.norm(y) < (1 + 5.0 / (dim + 1)) * hs * math.sqrt(dim) * sigma:
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * y

                # Update pc
                hsig_threshold = 1.5 * math.sqrt(dim) * sigma
                if np.linalg.norm(ps) < hsig_threshold:
                    pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * ps / sigma

                # Update C
                norm_ps_sq = np.dot(ps, ps)
                delta_h = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (norm_ps_sq / dim - 1)

                rank1 = np.outer(y, y)

                # Rank-mu update
                if len(valid_idx) >= mu:
                    Y_selected = Y_sorted[:mu]
                else:
                    Y_selected = Y[valid_idx]
                    while len(Y_selected) < mu:
                        Y_selected = np.vstack([Y_selected, m_old])

                Y_steps = (Y_selected[:mu] - m_old[None, :]) / max(sigma, 1e-12)
                rank_mu = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', Y_steps, Y_steps), axis=0)

                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
                C = (C + C.T) / 2.0
                update_eig()

                # Update sigma
                sigma = sigma * math.exp(cs / dcor * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

                # Convergence check
                if sigma < 1e-10:
                    break

        # Run CMA-ES on top 3 candidates from Phase 1
        n_cma_starts = min(3, len(evaluated_points))
        for i in range(n_cma_starts):
            if calls_made >= budget or time_left() < 10:
                break
            sigma_init = 0.5 if i == 0 else 0.3
            cma_es(evaluated_points[i].copy(), budget - calls_made, sigma_init)

        return best_x, best_f
# EVOLVE-BLOCK-END
