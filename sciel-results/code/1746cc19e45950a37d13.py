"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + (1+1)-ES for BBOB f20 (Schwefel).

    Strategy:
      1. Quick grid-based exploration to find promising basins.
      2. Nelder-Mead simplex from best grid points for fast convergence.
      3. Final (1+1)-ES polish with 1/5-rule adaptation for fine-tuning.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng()

        call_count = [0]

        def make_call(x):
            if call_count[0] >= total_budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(start_x, max_iter):
            """Run Nelder-Mead simplex optimization."""
            nonlocal call_count, best_x, best_f

            if time.time() > deadline - 5:
                return

            # Initialize simplex
            simplex = []
            f_start = make_call(start_x)
            if f_start is None:
                return
            update_best(start_x, f_start)
            simplex.append((start_x.copy(), f_start))

            # Other points of the simplex
            for i in range(dim):
                if time.time() > deadline - 5:
                    return
                if call_count[0] >= total_budget - 5:
                    return

                x_i = start_x.copy()
                step = 0.5
                if x_i[i] + step > 5.0:
                    step = -0.5
                x_i[i] += step
                x_i = np.clip(x_i, -5.0, 5.0)

                f_i = make_call(x_i)
                if f_i is None:
                    return
                update_best(x_i, f_i)
                simplex.append((x_i.copy(), f_i))

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            for _ in range(max_iter):
                if call_count[0] >= total_budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda x: x[1])

                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_best = simplex[0][0]
                f_best = simplex[0][1]

                # Check for convergence
                if f_worst - f_best < 1e-10:
                    break

                # Calculate centroid of all but worst point
                x_centroid = np.mean([x for x, _ in simplex[:-1]], axis=0)

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = make_call(x_ref)
                if f_ref is None:
                    break
                update_best(x_ref, f_ref)

                if f_ref < f_best:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = make_call(x_exp)
                    if f_exp is None:
                        break
                    update_best(x_exp, f_exp)

                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                elif f_ref < simplex[-2][1]:
                    simplex[-1] = (x_ref, f_ref)
                else:
                    # Contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        # Inside contraction
                        x_con = x_centroid + rho * (x_worst - x_centroid)

                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = make_call(x_con)
                    if f_con is None:
                        break
                    update_best(x_con, f_con)

                    if f_con < min(f_ref, f_worst):
                        simplex[-1] = (x_con, f_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if time.time() > deadline - 5:
                                return
                            if call_count[0] >= total_budget - 5:
                                return

                            x_shrink = x_best + sigma_shrink * (simplex[i][0] - x_best)
                            x_shrink = np.clip(x_shrink, -5.0, 5.0)
                            f_shrink = make_call(x_shrink)
                            if f_shrink is None:
                                return
                            update_best(x_shrink, f_shrink)
                            simplex[i] = (x_shrink, f_shrink)

                        # Check if simplex is too small
                        size = np.linalg.norm(simplex[-1][0] - x_best)
                        if size < 1e-7:
                            break

        def run_es(x_init, budget, sigma_start=1.0):
            """Run (1+1)-ES with 1/5-rule adaptation."""
            nonlocal call_count, best_x, best_f

            x = x_init.copy()
            fx = make_call(x)
            if fx is None:
                return x, fx
            update_best(x, fx)

            sigma = sigma_start
            min_sigma = 1e-8
            max_sigma = 5.0
            success_count = 0
            window = 10

            for _ in range(budget):
                if time.time() > deadline - 5:
                    break
                if call_count[0] >= total_budget - 5:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    success_count += 1
                else:
                    success_count += 0

                # 1/5-rule adaptation
                if success_count >= window:
                    if success_count >= window * 0.2:  # 1/5 success rate
                        sigma = min(sigma * 1.2, max_sigma)
                    else:
                        sigma = max(sigma * 0.8, min_sigma)
                    success_count = 0

            return x, fx

        # Phase 1: Quick grid exploration
        # Sample a coarse grid to find promising regions
        n_grid = 3
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        for m in range(n_grid):
                            x = np.array([
                                -4 + 8 * i / (n_grid - 1),
                                -4 + 8 * j / (n_grid - 1),
                                -4 + 8 * k / (n_grid - 1),
                                -4 + 8 * l / (n_grid - 1),
                                -4 + 8 * m / (n_grid - 1)
                            ])
                            grid_points.append(x)

        # Also add center and random points
        grid_points.append(np.zeros(dim))
        for _ in range(5):
            grid_points.append(rng.uniform(-4, 4, dim))

        # Shuffle
        rng.shuffle(grid_points)

        # Evaluate grid points (limit to 30 points max)
        grid_results = []
        for x in grid_points[:30]:
            if time.time() > deadline - 50:
                break
            if call_count[0] >= total_budget - 50:
                break

            f = make_call(x)
            if f is None:
                break
            update_best(x, f)
            grid_results.append((x, f))

        # Sort by function value and take top 3
        grid_results.sort(key=lambda x: x[1])
        top_points = [x for x, f in grid_results[:3]]

        # Phase 2: Nelder-Mead from top grid points
        nm_budget = 60
        for x_start in top_points:
            if time.time() > deadline - 30:
                break
            if call_count[0] >= total_budget - 30:
                break

            nelder_mead(x_start, nm_budget)

        # Phase 3: Final (1+1)-ES polish
        if best_x is not None and time.time() < deadline - 10 and call_count[0] < total_budget - 10:
            remaining = total_budget - call_count[0] - 5
            if remaining > 10:
                run_es(best_x, remaining, sigma_start=0.3)

        return best_x, best_f
# EVOLVE-BLOCK-END
