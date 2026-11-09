"""Optimized program for BBOB f21 (Gallagher 101 peaks) using CMA-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES with restarts for multimodal continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Global scan to find good starting points ---
        # Use a larger fraction of budget for global search on multimodal f21
        n_scan = min(60, budget // 10)
        if n_scan < 15:
            n_scan = 15

        # Generate initial points: mix of Latin hypercube and random
        points = []
        n_lh = n_scan // 2
        for i in range(n_lh):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = ((i + (d + 0.5) / n_lh) % 1.0) * 10.0 - 5.0
            perm = rng.permutation(dim)
            x = x[perm]
            points.append(x)

        for i in range(n_scan - n_lh):
            x = rng.uniform(-5, 5, dim)
            points.append(x)

        # Evaluate all scan points
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []
        for x in points:
            fx = func(np.clip(x, -5, 5))
            all_points.append(x.copy())
            all_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5)

        calls_made = n_scan

        # --- Phase 2: CMA-ES with restarts ---
        remaining = budget - n_scan
        if remaining <= 0:
            return best_x, best_f

        best_x_cma = best_x.copy()
        best_f_cma = best_f

        # Track best points found during scan for restarts
        # Sort by fitness and keep top few diverse points
        sorted_idx = np.argsort(all_fvals)
        top_points = [all_points[i].copy() for i in sorted_idx[:min(5, len(all_points))]]
        top_fvals = [all_fvals[i] for i in sorted_idx[:min(5, len(all_points))]]

        # CMA-ES parameters
        lam = max(4 + int(3 * np.log(dim)), 10)
        # Adjust lam to fit budget
        max_iter_per_run = remaining // lam
        if max_iter_per_run < 3:
            lam = max(4, remaining // 3)
            max_iter_per_run = remaining // lam

        # Weights and parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        def run_cma(start_mean, start_sigma, max_calls):
            """Run CMA-ES starting from start_mean with start_sigma, using up to max_calls."""
            nonlocal calls_made, best_x_cma, best_f_cma

            mean = start_mean.copy()
            sigma = start_sigma
            p_sigma = np.zeros(dim)
            C = np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(C)

            local_best_x = start_mean.copy()
            local_best_f = np.inf

            it = 0
            no_improve = 0
            threshold_iter = 15  # restart if no improvement for this many iterations

            while calls_made < max_calls and it < max_iter_per_run:
                # Get eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
                B_d = B * D

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = B_d @ z
                    x_new = np.clip(mean + sigma * y, -5, 5)
                    offspring.append(x_new)

                # Evaluate offspring
                fitnesses = []
                for x_new in offspring:
                    if calls_made >= max_calls:
                        fitnesses.append(np.inf)
                    else:
                        f_val = func(x_new)
                        fitnesses.append(f_val)
                        calls_made += 1

                fitnesses = np.array(fitnesses)
                order = np.argsort(fitnesses)
                y_sorted = [offspring[i] for i in order]
                f_sorted = fitnesses[order]

                # Update local best
                if f_sorted[0] < local_best_f:
                    local_best_f = f_sorted[0]
                    local_best_x = y_sorted[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Update global best
                if f_sorted[0] < best_f_cma:
                    best_f_cma = f_sorted[0]
                    best_x_cma = y_sorted[0].copy()

                # Update mean
                y_weighted = np.zeros(dim)
                for i in range(mu):
                    y_weighted += weights[i] * (y_sorted[i] - mean) / sigma
                mean_old = mean.copy()
                mean = mean + sigma * y_weighted
                mean = np.clip(mean, -5, 5)

                # Update sigma
                threshold = (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) * 1.4
                h_sigma = 1.0 if np.linalg.norm(y_weighted) / np.sqrt(mu_eff) < threshold else 0.0

                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_weighted
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                y_norm = (mean - mean_old) / sigma
                rank1 = np.outer(y_norm, y_norm)

                y_mu = np.zeros((dim, mu))
                for i in range(mu):
                    y_mu[:, i] = (y_sorted[i] - mean_old) / sigma
                rank_mu = np.outer(y_mu, weights[None, :] * y_mu)

                C = (1 + c_1 * (c_mu - c_1) + c_mu) * C \
                    - c_1 * rank1 \
                    - c_mu * rank_mu

                # Ensure positive definiteness
                eigvals_new, eigvecs_new = np.linalg.eigh(C)
                eigvals_new = np.maximum(eigvals_new, 1e-12)
                C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T

                it += 1

                # Check for stagnation
                if no_improve >= threshold_iter and it > 5:
                    break

                # Check if sigma is too small
                if sigma < 1e-6:
                    break

            return local_best_x, local_best_f, calls_made

        # Run CMA-ES with restarts
        # First run from best scan point
        start_sigma = 2.0
        local_x, local_f, _ = run_cma(best_x.copy(), start_sigma, budget)

        # If we have budget left, try restarts from other good points
        restart_attempts = 0
        max_restarts = 3

        while calls_made < budget - lam and restart_attempts < max_restarts:
            # Pick next restart point from top scan points
            if restart_attempts < len(top_points):
                start_point = top_points[restart_attempts].copy()
            else:
                # Perturb the best found point
                start_point = best_x_cma + rng.normal(0, 1.0, dim)
                start_point = np.clip(start_point, -5, 5)

            # Only restart if this point is not too close to the current best
            if np.linalg.norm(start_point - best_x_cma) < 0.5:
                restart_attempts += 1
                continue

            # Run CMA-ES from this point
            local_x, local_f, _ = run_cma(start_point, 1.5, budget)
            restart_attempts += 1

        # --- Phase 3: Final local refinement with remaining budget ---
        if calls_made < budget - 5:
            # Use remaining budget for a final Nelder-Mead-like local search
            # around the best point found
            nm_points = [best_x_cma.copy()]
            for d in range(dim):
                p_plus = best_x_cma.copy()
                p_plus[d] += 0.1
                p_minus = best_x_cma.copy()
                p_minus[d] -= 0.1
                nm_points.append(np.clip(p_plus, -5, 5))
                nm_points.append(np.clip(p_minus, -5, 5))

            nm_fvals = []
            for p in nm_points:
                if calls_made >= budget:
                    break
                f_val = func(p)
                nm_fvals.append(f_val)
                calls_made += 1

            if nm_fvals:
                nm_points = np.array(nm_points[:len(nm_fvals)])
                nm_fvals = np.array(nm_fvals)
                order = np.argsort(nm_fvals)
                nm_points = nm_points[order]
                nm_fvals = nm_fvals[order]

                # Simple local refinement
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                for _ in range(min(10, budget - calls_made)):
                    if calls_made >= budget:
                        break
                    # Reflection
                    centroid = np.mean(nm_points[1:], axis=0)
                    x_reflect = centroid + alpha * (centroid - nm_points[0])
                    x_reflect = np.clip(x_reflect, -5, 5)
                    f_reflect = func(x_reflect)
                    calls_made += 1

                    if f_reflect < nm_fvals[-1]:
                        nm_points[-1] = x_reflect
                        nm_fvals[-1] = f_reflect
                    elif f_reflect < nm_fvals[0]:
                        # Expansion
                        x_expand = centroid + gamma * (x_reflect - centroid)
                        x_expand = np.clip(x_expand, -5, 5)
                        f_expand = func(x_expand)
                        calls_made += 1
                        if f_expand < f_reflect:
                            nm_points[0] = x_expand
                            nm_fvals[0] = f_expand
                        else:
                            nm_points[0] = x_reflect
                            nm_fvals[0] = f_reflect
                    else:
                        # Contraction
                        x_contract = centroid + 0.5 * (nm_points[0] - centroid)
                        x_contract = np.clip(x_contract, -5, 5)
                        f_contract = func(x_contract)
                        calls_made += 1
                        if f_contract < nm_fvals[0]:
                            nm_points[0] = x_contract
                            nm_fvals[0] = f_contract
                        else:
                            # Shrink
                            for i in range(1, len(nm_points)):
                                nm_points[i] = nm_points[0] + rho * (nm_points[i] - nm_points[0])
                                nm_fvals[i] = func(nm_points[i])
                                calls_made += 1
                                if calls_made >= budget:
                                    break

                    # Re-sort
                    order = np.argsort(nm_fvals)
                    nm_points = nm_points[order]
                    nm_fvals = nm_fvals[order]

                    if nm_fvals[0] < best_f_cma:
                        best_f_cma = nm_fvals[0]
                        best_x_cma = nm_points[0].copy()

        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
