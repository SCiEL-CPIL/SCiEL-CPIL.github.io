"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid search exploiting Rastrigin structure + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Coarse-to-fine structured grid search ---
        # Rastrigin: f(x) = 10*A + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The function has periodic structure with period 2 in each dimension.
        # We exploit this by doing a coarse grid that samples every ~1 unit,
        # then progressively refine around the best point.

        # Step 1: Very coarse grid - sample every 2.0 units (period of cos)
        # This gives us 5 points per dimension = 3125 points, too many.
        # Instead, use a Latin hypercube-like approach with structured offsets.

        # Actually, let's do a smarter approach:
        # 1) Sample a moderate grid to find the basin
        # 2) Do coordinate-wise 1D scans to refine each coordinate

        # Phase 1a: Initial coarse sampling using a stratified grid
        # Use 3 levels per dimension: -4, 0, 4 (within [-4,4] where minimizer lies)
        # Plus some intermediate points for better coverage
        coarse_points = []
        # Grid with spacing ~2.5 in [-4, 4]
        levels = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        # Generate all combinations (5^5 = 3125) - too many, so we'll use a subset
        # Use a space-filling sequence over the grid

        # Better approach: Use a deterministic space-filling sequence
        # that covers the domain well, then do local refinement.

        # Let's use a different strategy:
        # 1) Evaluate a set of well-distributed points (Latin hypercube style)
        # 2) Identify the best region
        # 3) Do coordinate-wise 1D optimization (golden section or ternary search)
        # 4) Polish with Nelder-Mead

        # Phase 1: Latin Hypercube sampling
        n_lhs = min(100, budget // 5)  # 100 points for LHS
        rng = np.random.RandomState(42)

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Divide [0,1] into n_lhs equal intervals, pick one random point in each
            intervals = np.arange(n_lhs) / n_lhs
            rng.shuffle(intervals)
            lhs_points[:, d] = -5.0 + 10.0 * (intervals + rng.random(n_lhs) / n_lhs)

        for i in range(n_lhs):
            if not time_ok() or calls_made >= budget:
                break
            try_point(lhs_points[i])

        # Phase 2: Coordinate-wise 1D refinement
        # For each coordinate, do a ternary/golden section search
        # This exploits the separable structure of Rastrigin

        if best_x is not None and time_ok() and calls_made < budget:
            # Multiple passes of coordinate-wise refinement
            for pass_num in range(3):
                if not time_ok() or calls_made >= budget:
                    break
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    # Golden section search along dimension d
                    a = max(-5.0, best_x[d] - 2.0)
                    b = min(5.0, best_x[d] + 2.0)

                    # Golden ratio
                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                    c = b - gr * (b - a)
                    d_val = a + gr * (b - a)

                    # Evaluate at c and d
                    x_c = best_x.copy()
                    x_c[d] = c
                    x_d = best_x.copy()
                    x_d[d] = d_val

                    if not time_ok() or calls_made >= budget:
                        break
                    try_point(x_c)
                    if not time_ok() or calls_made >= budget:
                        break
                    try_point(x_d)

                    # Get function values (we need to track them)
                    # Since try_point updates best, we need a different approach
                    # Let's do a simple coordinate-wise line search with explicit tracking

                    # Actually, let's do a simpler but effective approach:
                    # For each coordinate, try a few candidate values around the current best
                    # and keep the one that gives the lowest function value

                    # Reset for this coordinate search
                    local_best_x = best_x.copy()
                    local_best_f = best_f

                    # Try values in a range around current best
                    for offset in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, -2.0, 2.0]:
                        if not time_ok() or calls_made >= budget:
                            break
                        x_try = local_best_x.copy()
                        x_try[d] = local_best_x[d] + offset
                        x_try = np.clip(x_try, -5.0, 5.0)
                        fx = func(x_try)
                        calls_made += 1
                        if fx < local_best_f:
                            local_best_f = fx
                            local_best_x = x_try.copy()

                    # Update global best
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

        # Phase 3: Finer coordinate-wise refinement
        if best_x is not None and time_ok() and calls_made < budget:
            for pass_num in range(2):
                if not time_ok() or calls_made >= budget:
                    break
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    local_best_x = best_x.copy()
                    local_best_f = best_f

                    # Finer offsets
                    for offset in [-0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5]:
                        if not time_ok() or calls_made >= budget:
                            break
                        x_try = local_best_x.copy()
                        x_try[d] = local_best_x[d] + offset
                        x_try = np.clip(x_try, -5.0, 5.0)
                        fx = func(x_try)
                        calls_made += 1
                        if fx < local_best_f:
                            local_best_f = fx
                            local_best_x = x_try.copy()

                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

        # Phase 4: Nelder-Mead polish
        if best_x is not None and time_ok() and calls_made < budget:
            def nm_minimize(func, x0, budget_left, deadline):
                dim = len(x0)
                simplex = [x0.copy()]
                f0 = func(np.clip(x0, -5, 5))
                simplex_f = [f0]
                calls = 1

                for i in range(dim):
                    x_i = x0.copy()
                    x_i[i] += 0.1
                    simplex.append(x_i)
                    f_i = func(np.clip(x_i, -5, 5))
                    simplex_f.append(f_i)
                    calls += 1

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                while calls < budget_left and time.time() < deadline:
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls += 1

                    if f_r < simplex_f[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        if f_r < simplex_f[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < min(f_r, simplex_f[-1]):
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                                calls += 1
                                if calls >= budget_left or time.time() > deadline:
                                    break
                            if calls >= budget_left or time.time() > deadline:
                                break

                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()

                    if calls >= budget_left:
                        break

                min_idx = np.argmin(simplex_f)
                return simplex[min_idx], simplex_f[min_idx], calls

            nm_budget = budget - calls_made
            if nm_budget > dim + 1:
                x_nm, f_nm, nm_calls = nm_minimize(func, best_x, nm_budget, deadline)
                calls_made += nm_calls
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
