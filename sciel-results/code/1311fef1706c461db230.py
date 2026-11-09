"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES for basin exploration + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # Phase 1: Multi-restart (1+1)-ES for basin exploration
        # Use ~60% of budget for exploration
        exploration_budget = int(budget * 0.6)
        n_restarts = 6
        budget_per_restart = exploration_budget // n_restarts

        # Collect diverse starting points and their best results
        candidates = []  # list of (x, f)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Diverse initial points
            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-4, 4, dim)
            elif restart == 2:
                x = rng.uniform(-4, 4, dim)
            elif restart == 3:
                x = rng.uniform(-3, 3, dim)
            elif restart == 4:
                x = rng.uniform(-5, 5, dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            candidates.append((x.copy(), fx))

            # (1+1)-ES with 1/5 success rule
            sigma = 2.0
            evals_in_restart = 1
            no_improve = 0
            max_no_improve = 20  # Early termination if stuck

            for _ in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                if evals_in_restart >= budget_per_restart:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evals_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break

                # Keep sigma within reasonable bounds
                if sigma < 1e-3:
                    sigma = 1e-3
                if sigma > 8.0:
                    sigma = 8.0

            candidates.append((x.copy(), fx))

        # Phase 2: CMA-ES refinement of the best candidates
        # Use remaining ~40% of budget
        remaining_budget = budget - int(time.time() - start_time)  # approximate
        # Actually, let's use a fixed fraction
        refinement_budget = int(budget * 0.4)

        # Sort candidates by fitness and take top 3 for refinement
        candidates.sort(key=lambda t: t[1])
        top_candidates = candidates[:min(3, len(candidates))]

        for cx, cf in top_candidates:
            if time.time() > deadline:
                break
            if refinement_budget <= 0:
                break

            # CMA-ES parameters
            lam = 8
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = 2 / ((dim + 2)**2 + mu_eff) - c1
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            # Initialize CMA-ES from candidate
            mean = cx.copy()
            sigma = 0.5  # Smaller initial sigma for local refinement
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))

            evals_used = 0
            max_evals = refinement_budget // len(top_candidates)

            while evals_used < max_evals:
                if time.time() > deadline:
                    break

                # Generate population
                pop = np.zeros((lam, dim))
                fit = np.zeros(lam)

                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

                for i in range(lam):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = L @ z
                    xi = np.clip(mean + sigma * y, -5, 5)
                    pop[i] = xi
                    fit[i] = func(xi)
                    evals_used += 1

                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = xi.copy()

                if evals_used < lam:
                    break

                # Sort by fitness
                order = np.argsort(fit)
                fit_sorted = fit[order]
                pop_sorted = pop[order]

                # Update mean
                mean_old = mean.copy()
                mean = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)

                # Update evolution paths
                diff = mean - mean_old
                inv_sigma = 1.0 / sigma
                ps = (1 - cs) * ps + cs * inv_sigma * (L.T @ diff)

                pc = (1 - cc) * pc + cc * cs / (1 - cs) * (
                    (1 + cs) * inv_sigma * diff - ps
                )

                # Update covariance
                C += c1 * (np.outer(diff, diff) - C)

                diff_mu = pop_sorted[:mu] - mean_old
                rank_mu = np.zeros((dim, dim))
                for k in range(mu):
                    rank_mu += weights[k] * np.outer(diff_mu[k], diff_mu[k])
                C += cmu * rank_mu

                # Correction for pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (evals_used // lam + 1)))) < (1.4 + 2/(dim+1)) * chiN
                if hsig:
                    C += (c1 * (hsig - 1 + cc) + cmu * (hsig - 1)) * (cc / (1 - cc) * C + pc)

                # Symmetrize
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))

                # Keep sigma within reasonable bounds
                if sigma < 1e-4:
                    sigma = 1e-4
                if sigma > 5.0:
                    sigma = 5.0

                # Re-orthogonalize periodically
                if evals_used % (lam * 3) < lam:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-8)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
