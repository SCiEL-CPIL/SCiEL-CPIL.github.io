import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Global exploration with random sampling
        n_explore = min(30, budget)
        x_samples = rng.uniform(-5, 5, (n_explore, dim))
        f_samples = np.array([func(x) for x in x_samples])
        calls = n_explore

        # Pick the best few as starting points for local optimization
        # Use top k diverse points
        k_starts = 5
        # Sort by fitness
        order = np.argsort(f_samples)
        # Pick top k but ensure diversity
        starts = []
        for idx in order:
            if len(starts) >= k_starts:
                break
            # Check if this point is sufficiently different from existing starts
            if not starts:
                starts.append(x_samples[idx].copy())
            else:
                min_dist = min(np.linalg.norm(x_samples[idx] - s) for s in starts)
                if min_dist > 1.0:  # diversity threshold
                    starts.append(x_samples[idx].copy())
        
        # If we didn't get enough diverse points, just use the top ones
        while len(starts) < k_starts:
            starts.append(x_samples[order[len(starts) % n_explore]].copy())

        x_ever = x_samples[order[0]].copy()
        f_ever = f_samples[order[0]]

        # Phase 2: Local optimization with Nelder-Mead on each start
        # Distribute remaining budget among starts
        remaining_budget = budget - calls
        budget_per_start = max(10, remaining_budget // k_starts)

        for s_idx, start_x in enumerate(starts):
            if not time_ok():
                break
            if calls >= budget:
                break

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Initial simplex
            simplex = [start_x.copy()]
            for i in range(dim):
                dx = start_x.copy()
                step = 0.1 if abs(start_x[i]) > 0.1 else 0.05
                dx[i] += step
                dx = np.clip(dx, -5, 5)
                simplex.append(dx.copy())

            f_simplex = np.array([func(x) for x in simplex])
            calls += dim

            if f_simplex[0] < f_ever:
                f_ever = f_simplex[0]
                x_ever = simplex[0].copy()

            # Nelder-Mead iterations
            max_iters = budget_per_start // (dim + 1)
            for it in range(max_iters):
                if not time_ok() or calls >= budget:
                    break

                # Sort simplex by function value
                order_sm = np.argsort(f_simplex)
                simplex = [simplex[i].copy() for i in order_sm]
                f_simplex = f_simplex[order_sm]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5, 5)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1

                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp.copy()
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref.copy()
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-1]:
                    # Accept reflection
                    simplex[-1] = x_ref.copy()
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_simplex[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    if f_con < min(f_ref, f_simplex[-1]):
                        simplex[-1] = x_con.copy()
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break

                # Track best
                best_idx = np.argmin(f_simplex)
                if f_simplex[best_idx] < f_ever:
                    f_ever = f_simplex[best_idx]
                    x_ever = simplex[best_idx].copy()

                # Check for convergence
                f_range = np.max(f_simplex) - np.min(f_simplex)
                if f_range < 1e-8:
                    break

        return x_ever, f_ever
