"""Portfolio search optimizer for BBOB f20 (Schwefel) with multi-phase refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size, Nelder-Mead polish,
    and coordinate descent. Designed for BBOB f20 (Schwefel) on [-5,5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # --- Phase 1: Multi-start (1+1)-ES with diverse initializations ---
        # Generate diverse starting points
        n_starts = 12
        starts = []
        # Structured points for coverage
        structured = [
            np.zeros(dim),
            2.0 * np.ones(dim),
            -2.0 * np.ones(dim),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0])[:dim],
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0])[:dim],
            np.array([3.0, 3.0, -3.0, -3.0, 3.0])[:dim],
            np.array([-3.0, -3.0, 3.0, 3.0, -3.0])[:dim],
        ]
        for pt in structured:
            starts.append(pt.copy())
        # Random points
        for i in range(n_starts - len(starts)):
            rng_local = np.random.default_rng(555 + i * 211)
            starts.append(rng_local.uniform(-4.5, 4.5, dim))

        # Budget allocation: early starts get more, later starts get less
        # Reserve 60 calls for polish phases
        polish_reserve = 60
        search_budget = total_budget - polish_reserve

        budget_allocs = []
        remaining = search_budget
        for i in range(n_starts):
            if i < 4:
                alloc = max(35, remaining // 4)
            elif i < 8:
                alloc = max(25, remaining // 8)
            else:
                alloc = max(15, remaining // 12)
            alloc = min(alloc, remaining)
            budget_allocs.append(alloc)
            remaining -= alloc

        for start_idx, x0 in enumerate(starts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            alloc = budget_allocs[start_idx]
            if alloc <= 0:
                continue

            rng = np.random.default_rng(777 + start_idx * 104729)

            x = x0.copy()
            fx = try_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.8
            sigma_min = 1e-8
            sigma_max = 4.5

            successes = 0
            steps = 0
            no_improve = 0
            last_good_sigma = sigma

            for _ in range(alloc):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break

                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    sigma = min(sigma * 1.25, sigma_max)
                    last_good_sigma = sigma
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.82, sigma_min)

                # Success-rate-based adaptation
                if steps >= 12:
                    sr = successes / steps
                    if sr > 0.38:
                        sigma = min(sigma * 1.12, sigma_max)
                    elif sr < 0.08:
                        sigma = max(sigma * 0.88, sigma_min)
                    if steps >= 60:
                        successes = 0
                        steps = 0

                # Stagnation escape: larger jump
                if no_improve > 25:
                    sigma = min(max(sigma * 2.0, last_good_sigma * 1.5), sigma_max)
                    no_improve = 0

        # --- Phase 2: Nelder-Mead polish from best point ---
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            from scipy.optimize import minimize as sp_minimize

            nm_calls = [0]
            nm_budget = min(total_budget - calls_made, 50)

            def nm_func(x_vec):
                if nm_calls[0] >= nm_budget or time.time() > deadline:
                    return best_f + 1e10
                val = func(np.clip(x_vec, -5, 5))
                nm_calls[0] += 1
                calls_made += 1
                return val

            try:
                res = sp_minimize(
                    nm_func, best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # --- Phase 3: Coordinate descent with adaptive step sizes ---
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            rng_cd = np.random.default_rng(9999)
            x = best_x.copy()
            fx = best_f
            cd_remaining = total_budget - calls_made
            base_step = 0.5

            for cd_iter in range(min(cd_remaining, 40)):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                # Try each coordinate
                improved_this_pass = False
                for d in range(dim):
                    if calls_made >= total_budget or time.time() > deadline:
                        break

                    step = base_step * (0.5 ** cd_iter)
                    step = max(step, 1e-6)

                    # Try positive and negative directions
                    for sign in [1.0, -1.0]:
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            break
                        if fy < fx:
                            x = y
                            fx = fy
                            improved_this_pass = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            # Shrink step on success
                            step *= 0.5
                        else:
                            # If no improvement, try smaller step
                            step2 = step * 0.3
                            if step2 > 1e-7:
                                y2 = x.copy()
                                y2[d] += sign * step2
                                y2 = np.clip(y2, -5, 5)
                                fy2 = try_func(y2)
                                if fy2 is None:
                                    break
                                if fy2 < fx:
                                    x = y2
                                    fx = fy2
                                    improved_this_pass = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()

                if not improved_this_pass:
                    # No coordinate improved; try a small random perturbation
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    y = np.clip(x + 0.05 * rng_cd.standard_normal(dim), -5, 5)
                    fy = try_func(y)
                    if fy is None:
                        break
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Stuck; break out
                        break

        # --- Phase 4: Final fine ES polish ---
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            rng_fin = np.random.default_rng(4242)
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            fin_remaining = min(total_budget - calls_made, 15)

            for _ in range(fin_remaining):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng_fin.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = max(sigma * 0.85, 1e-9)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, 1e-9)

        # Fallback
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
