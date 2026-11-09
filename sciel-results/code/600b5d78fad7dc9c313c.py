"""Rewritten program: coordinate-descent variant of Nelder-Mead for Rosenbrock valley."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Valley-adaptive coordinate descent with Nelder-Mead refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 0: Multi-start initialization ---
        # Try several diverse starting points to handle hidden shift/rotation
        candidates = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]) if dim == 5 else np.ones(dim) * 2.0,
            np.array([3.0, 1.5, 0.5, -1.0, -2.0]) if dim == 5 else rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf
        calls = 0

        for c in candidates:
            if calls >= budget:
                break
            c_clipped = np.clip(c, -5, 5)
            fc = func(c_clipped)
            calls += 1
            if fc < best_f:
                best_f = fc
                best_x = c_clipped.copy()

        x = best_x.copy()
        fx = best_f
        x_best = x.copy()
        fx_best = fx

        # --- Phase 1: Adaptive coordinate descent with valley-aware steps ---
        # Rosenbrock has a narrow valley; use larger steps along correlated directions
        # and adaptively shrink when stuck.

        # Initialize step sizes: larger for early coordinates (valley alignment heuristic)
        steps = np.ones(dim) * 1.5
        min_step = 1e-7
        max_step = 15.0
        alpha = 1.2   # growth on success
        beta = 0.6    # shrink on failure
        stall_count = 0
        max_stall = 8

        # Track per-coordinate success rates to bias step sizes
        coord_success = np.ones(dim) / dim
        coord_tries = np.ones(dim)

        while calls < budget:
            improved_this_pass = False

            for i in range(dim):
                if calls >= budget:
                    break

                # Adaptive step: larger if this coord has been successful
                step_i = steps[i]

                for sign in [1.0, -1.0]:
                    if calls >= budget:
                        break

                    x_trial = x.copy()
                    x_trial[i] += sign * step_i
                    x_trial = np.clip(x_trial, -5, 5)

                    if np.allclose(x_trial, x, atol=1e-12):
                        continue

                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < fx:
                        # Accept the move
                        x = x_trial
                        fx = f_trial
                        steps[i] = min(steps[i] * alpha, max_step)
                        improved_this_pass = True
                        coord_success[i] = (coord_success[i] * coord_tries[i] + 1) / (coord_tries[i] + 1)
                        coord_tries[i] += 1
                        if fx < fx_best:
                            x_best = x.copy()
                            fx_best = fx
                        break  # Move to next coordinate
                    else:
                        steps[i] = max(steps[i] * beta, min_step)
                        coord_tries[i] += 1

            # Stall detection
            if not improved_this_pass:
                stall_count += 1
                # Shrink all steps
                steps *= 0.7
                if np.all(steps < min_step):
                    break

                # If stalled too long, try a random kick to escape
                if stall_count >= max_stall and calls < budget - 5:
                    kick_size = max(np.min(steps) * 2, 0.1)
                    x_kick = x + rng.uniform(-1, 1, dim) * kick_size
                    x_kick = np.clip(x_kick, -5, 5)
                    f_kick = func(x_kick)
                    calls += 1
                    if f_kick < fx:
                        x = x_kick
                        fx = f_kick
                        if fx < fx_best:
                            x_best = x.copy()
                            fx_best = fx
                    stall_count = 0
                    steps = np.clip(steps * 1.5, min_step, max_step)
            else:
                stall_count = 0

        # --- Phase 2: Nelder-Mead simplex refinement ---
        # Build a simplex around the best point found so far
        if calls < budget - dim - 2:
            # Create simplex
            simplex = [x_best.copy()]
            base_step = 0.5
            for i in range(dim):
                if calls >= budget - 2:
                    break
                v = x_best.copy()
                v[i] += base_step
                v = np.clip(v, -5, 5)
                fv = func(v)
                calls += 1
                simplex.append(v)

            # Evaluate all vertices
            f_simplex = [fx_best]
            for v in simplex[1:]:
                f_simplex.append(func(v) if calls < budget else np.inf)
                calls += 1

            # Nelder-Mead iterations
            alpha_nm = 1.0   # reflection
            gamma_nm = 2.0   # expansion
            rho_nm = 0.5     # contraction
            sigma_nm = 0.5   # shrink

            nm_iter = 0
            max_nm_iter = min(budget - calls, 100)

            while nm_iter < max_nm_iter and calls < budget - 2:
                nm_iter += 1

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflect
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                if np.allclose(x_ref, centroid, atol=1e-14):
                    break
                f_ref = func(x_ref)
                calls += 1

                if f_ref < f_simplex[0]:
                    # Try expansion
                    x_exp = centroid + gamma_nm * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    if np.allclose(x_exp, centroid, atol=1e-14):
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                    else:
                        f_exp = func(x_exp)
                        calls += 1
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_simplex[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Try contraction
                    x_con = centroid + rho_nm * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if np.allclose(x_con, centroid, atol=1e-14):
                        break
                    f_con = func(x_con)
                    calls += 1
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1

                # Update best
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < fx_best:
                    fx_best = f_simplex[min_idx]
                    x_best = simplex[min_idx].copy()

        # --- Phase 3: Final coordinate polish ---
        if calls < budget:
            steps = np.ones(dim) * 0.05
            polish_iters = min(30, budget - calls)
            for _ in range(polish_iters):
                if calls >= budget:
                    break
                for i in range(dim):
                    if calls >= budget:
                        break
                    for sign in [1.0, -1.0]:
                        if calls >= budget:
                            break
                        x_trial = x_best.copy()
                        x_trial[i] += sign * steps[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        if np.allclose(x_trial, x_best, atol=1e-14):
                            continue
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < fx_best:
                            x_best = x_trial
                            fx_best = f_trial
                            break
                steps *= 0.7

        return x_best, fx_best
# EVOLVE-BLOCK-END
