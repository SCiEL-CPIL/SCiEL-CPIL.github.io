"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid PSO + (mu+lambda)-ES with Rastrigin-aware coordinate refinement."""

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

        # --- Phase 1: PSO for broad exploration ---
        n_particles = 10
        w = 0.7
        c1 = 1.49445
        c2 = 1.49445
        v_max = 2.0

        # Initialize particles
        positions = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(n_particles)])
        velocities = np.zeros((n_particles, dim))
        p_f = np.array([eval_func(p) for p in positions])
        p_best = positions.copy()
        p_best_f = p_f.copy()
        g_best = p_best[np.argmin(p_best_f)].copy()
        g_best_f = p_best_f.min()

        pso_generations = 0
        no_improve_pso = 0
        g_best_prev = g_best_f

        while budget_ok() and time_ok() and call_count < int(budget * 0.5):
            pso_generations += 1

            for i in range(n_particles):
                if not budget_ok() or not time_ok():
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (p_best[i] - positions[i]) +
                                 c2 * r2 * (g_best - positions[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5.0, 5.0)

                f_new = eval_func(positions[i])
                if f_new < p_best_f[i]:
                    p_best_f[i] = f_new
                    p_best[i] = positions[i].copy()
                    if f_new < g_best_f:
                        g_best_f = f_new
                        g_best = positions[i].copy()

            if g_best_f < g_best_prev - 1e-6:
                no_improve_pso = 0
                g_best_prev = g_best_f
            else:
                no_improve_pso += 1

            # Adapt PSO parameters
            if no_improve_pso >= 5:
                w = max(w * 0.9, 0.4)
                no_improve_pso = 0

        # --- Phase 2: (mu+lambda)-ES for fine-grained search ---
        mu = 4
        lam = 12
        sigma = 1.5

        # Initialize population around current best
        pop = np.array([np.clip(g_best + rng.standard_normal(dim) * 1.0, -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        # Include the global best
        pop = np.vstack([g_best.reshape(1, -1), pop])
        pop_f = np.concatenate([[g_best_f], pop_f])
        idx = np.argsort(pop_f)
        pop = pop[idx][:mu]
        pop_f = pop_f[idx][:mu]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f

        while budget_ok() and time_ok():
            generation += 1

            # Adaptive sigma based on success rate
            if generation % 10 == 0:
                if best_f < best_f_prev - 1e-6:
                    sigma = min(sigma * 1.3, 4.0)
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
            if generation % 10 == 0:
                x_ref, f_ref = local_refine(pop[0].copy(), pop_f[0], max_rounds=3)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref

            # Restart if stuck
            if sigma < 0.15 and generation % 20 == 0:
                for _ in range(2):
                    if not budget_ok() or not time_ok():
                        break
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                sigma = 1.5
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
