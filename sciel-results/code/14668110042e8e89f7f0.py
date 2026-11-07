"""Nelder-Mead simplex based optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start Nelder-Mead simplex optimizer for BBOB f20 (Schwefel) on [-5,5]^5.
    Uses geometric reflection/expansion/contraction to navigate rugged landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # Generate diverse starting points
        n_starts = 8
        starts = []

        # Structured points for good coverage
        structured = [
            np.zeros(dim),
            np.ones(dim) * 1.5,
            np.ones(dim) * -1.5,
            np.array([3.0, -3.0, 3.0, -3.0, 3.0])[:dim],
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0])[:dim],
        ]
        for pt in structured:
            starts.append(pt.copy())

        # Random points
        for i in range(n_starts - len(structured)):
            local_rng = np.random.default_rng(2000 + i * 97)
            starts.append(local_rng.uniform(-4.0, 4.0, dim))

        # Reserve budget for final polish
        polish_budget = 40
        remaining = total_budget - polish_budget

        # Allocate budget per start: early starts get more
        budget_allocs = []
        rem = remaining
        for i in range(n_starts):
            if i < 3:
                alloc = max(35, rem // 4)
            elif i < 6:
                alloc = max(25, rem // 8)
            else:
                alloc = max(15, rem // 12)
            alloc = min(alloc, rem)
            budget_allocs.append(alloc)
            rem -= alloc

        for start_idx, x0 in enumerate(starts):
            if calls_made >= total_budget or time.time() > deadline:
                break

            alloc = budget_allocs[start_idx]
            if alloc <= 0:
                continue

            # Initialize Nelder-Mead simplex
            # Initial simplex: x0 and x0 + delta*e_i for each dimension
            delta = 0.5
            simplex = [x0.copy()]
            for d in range(dim):
                v = x0.copy()
                v[d] += delta
                simplex.append(v)

            # Evaluate simplex vertices
            simplex_f = []
            for v in simplex:
                fv = try_func(np.clip(v, -5, 5))
                if fv is None:
                    break
                simplex_f.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = np.clip(v, -5, 5).copy()
            else:
                # Complete NM iterations
                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                max_iters = alloc - (dim + 1)

                for _ in range(max_iters):
                    if calls_made >= total_budget or time.time() > deadline:
                        break

                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Compute centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    worst = simplex[-1]
                    x_ref = centroid + alpha * (centroid - worst)
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = try_func(x_ref)
                    if f_ref is None:
                        break

                    # Check if reflection improved
                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = try_func(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < simplex_f[-1]:
                            # Outside contraction
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            # Inside contraction
                            x_con = centroid - rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = try_func(x_con)
                        if f_con is None:
                            break

                        if f_con < min(f_ref, simplex_f[-1]):
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_shrink = try_func(simplex[i])
                                if f_shrink is None:
                                    break
                                simplex_f[i] = f_shrink
                            else:
                                # Update best
                                for i in range(dim + 1):
                                    if simplex_f[i] < best_f:
                                        best_f = simplex_f[i]
                                        best_x = simplex[i].copy()
                                break
                            if f_shrink is None:
                                break

                    # Track global best
                    for i in range(dim + 1):
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()

        # Final polish: small Nelder-Mead around best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            delta_polish = 0.1
            simplex = [best_x.copy()]
            for d in range(dim):
                v = best_x.copy()
                v[d] += delta_polish
                simplex.append(v)

            simplex_f = []
            for v in simplex:
                fv = try_func(np.clip(v, -5, 5))
                if fv is None:
                    break
                simplex_f.append(fv)
                if fv < best_f:
                    best_f = fv
                    best_x = np.clip(v, -5, 5).copy()
            else:
                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                max_iters = polish_budget - (dim + 1)

                for _ in range(max_iters):
                    if calls_made >= total_budget or time.time() > deadline:
                        break

                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    centroid = np.mean(simplex[:-1], axis=0)
                    worst = simplex[-1]

                    x_ref = centroid + alpha * (centroid - worst)
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = try_func(x_ref)
                    if f_ref is None:
                        break

                    if f_ref < simplex_f[0]:
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = try_func(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        if f_ref < simplex_f[-1]:
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            x_con = centroid - rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = try_func(x_con)
                        if f_con is None:
                            break

                        if f_con < min(f_ref, simplex_f[-1]):
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_shrink = try_func(simplex[i])
                                if f_shrink is None:
                                    break
                                simplex_f[i] = f_shrink
                            else:
                                for i in range(dim + 1):
                                    if simplex_f[i] < best_f:
                                        best_f = simplex_f[i]
                                        best_x = simplex[i].copy()
                                break
                            if f_shrink is None:
                                break

                    for i in range(dim + 1):
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
