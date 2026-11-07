"""Optimized search for BBOB f15 (rotated Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (μ+λ)-ES with adaptive step, multi-restart,
    and axis-wise Rastrigin-periodic refinement for f15 (rotated Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count < budget

        # --- Local refinement: coordinate descent with Rastrigin snap ---
        def local_refine(x, fx, max_rounds=5):
            """Coordinate descent with Rastrigin-periodic candidates."""
            nonlocal call_count, best_x, best_f
            if not budget_ok() or not time_ok():
                return x, fx

            for _ in range(max_rounds):
                if not budget_ok() or not time_ok():
                    break
                improved = False
                for axis in range(dim):
                    if not budget_ok() or not time_ok():
                        break
                    x0 = x[axis]
                    # Generate candidates: current, nearby, and Rastrigin-aligned
                    candidates = [x0]
                    # Small perturbations
                    for d in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
                        candidates.append(x0 + d)
                    # Rastrigin: minima at multiples of 2*pi in the unrotated frame.
                    k = np.round(x0 / (2.0 * np.pi))
                    for kk in [k - 1, k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi)
                    # Also try halfway points (maxima of cos, so avoid)
                    for kk in [k, k + 1]:
                        candidates.append(kk * 2.0 * np.pi + np.pi)

                    best_cand = x0
                    best_cand_f = fx
                    for c in candidates:
                        if not budget_ok():
                            break
                        c = float(np.clip(c, -5.0, 5.0))
                        if abs(c - x0) < 1e-12:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                if not improved:
                    break
            return x, fx

        # --- Main search: (mu+lambda)-ES with periodic restarts ---
        # Phase 1: Broad exploration with population
        mu = 4
        lam = 12
        sigma = 2.5
        # Initial population
        pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        # Sort
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f

        while budget_ok() and time_ok():
            generation += 1

            # Adaptive sigma based on success rate (1/5 rule)
            if generation % 10 == 0:
                # Check if we've been improving
                if best_f < best_f_prev - 1e-6:
                    sigma = min(sigma * 1.3, 8.0)
                    best_f_prev = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        sigma = max(sigma * 0.7, 0.1)
                        no_improve_count = 0

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if not budget_ok() or not time_ok():
                    break
                # Select parent (tournament or uniform from best mu)
                parent_idx = rng.integers(0, mu)
                parent = pop[parent_idx]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Combine parents and offspring, select best mu
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            idx = np.argsort(all_f)[:mu]
            pop = all_x[idx]
            pop_f = all_f[idx]

            # Periodic local refinement on the best point
            if generation % 15 == 0:
                x_ref, f_ref = local_refine(pop[0].copy(), pop_f[0], max_rounds=3)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref

            # Restart if stuck: if sigma is very small and no improvement
            if sigma < 0.15 and generation % 30 == 0:
                # Restart with a few new random points
                for _ in range(2):
                    if not budget_ok() or not time_ok():
                        break
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    # Replace worst if better
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                sigma = 2.0
                no_improve_count = 0

        # Final phase: intensive local refinement on best point
        if best_x is not None and budget_ok() and time_ok():
            x_final = best_x.copy()
            f_final = best_f
            for _ in range(5):
                if not budget_ok() or not time_ok():
                    break
                x_ref, f_ref = local_refine(x_final.copy(), f_final, max_rounds=5)
                if f_ref < f_final:
                    f_final = f_ref
                    x_final = x_ref.copy()
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
