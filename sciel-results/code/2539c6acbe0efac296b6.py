import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 270  # Safety margin well below 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(12345)

        # ---- Helper: dominance check ----
        def dominates(a, b):
            # a dominates b if a <= b in all objectives and < in at least one
            return np.all(a <= b) and np.any(a < b)

        # ---- Initialization: diverse stratified + boundary sampling ----
        # Use a good chunk of budget for initialization to get a wide spread
        n_init = max(30, budget // 6)  # ~33 evals for budget=200

        # Generate stratified points in [0,1]^dim
        n_per_axis = max(2, int(round(n_init ** (1.0 / dim))))
        # Create a grid but subsample to get n_init points
        grid = np.meshgrid(*[np.linspace(0, 1, n_per_axis) for _ in range(dim)])
        grid_flat = np.array([g.ravel() for g in grid]).T  # shape (n_per_axis^dim, dim)

        if len(grid_flat) >= n_init:
            # Randomly sample n_init from the grid
            idx = rng.choice(len(grid_flat), size=n_init, replace=False)
            X_init = grid_flat[idx]
        else:
            X_init = grid_flat.copy()
            # Fill remaining with random
            while len(X_init) < n_init:
                X_init = np.vstack([X_init, rng.random((1, dim))])

        # Add boundary points (corners and edges) to help find Pareto extremes
        n_boundary = max(8, n_init // 4)
        boundary_pts = []
        # All corners
        for mask in range(1 << dim):
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = 1.0 if (mask >> d) & 1 else 0.0
            boundary_pts.append(pt)
        # Add some random edge points
        while len(boundary_pts) < n_boundary:
            pt = rng.random(dim)
            # Force 1-2 dims to be at boundary
            n_fix = rng.integers(1, min(3, dim + 1))
            fix_dims = rng.choice(dim, size=n_fix, replace=False)
            for d in fix_dims:
                pt[d] = rng.choice([0.0, 1.0])
            boundary_pts.append(pt)

        X_init = np.vstack([X_init, np.array(boundary_pts[:n_boundary])])
        X_init = X_init * span + lo
        X_init = np.clip(X_init, lo, hi)

        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used_evals = len(X_init)

        # ---- Build initial Pareto front ----
        # Store as list of (x, f) tuples
        pareto_front = []

        for i in range(len(X_init)):
            f = F_init[i]
            x = X_init[i]
            # Check if dominated by any existing front member
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, f):
                    dominated = True
                    break
            if not dominated:
                # Remove members dominated by f
                pareto_front = [(x_p, f_p) for x_p, f_p in pareto_front if not dominates(f, f_p)]
                pareto_front.append((x, f))

        # ---- Crowding distance for diversity preservation ----
        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max - f_min > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
                else:
                    cd[sorted_idx[1:-1]] = np.inf
            return cd

        # ---- Main loop: adaptive mutation with crowding-based selection ----
        sigma = 0.12  # initial mutation scale
        min_sigma = 0.003
        max_sigma = 0.25
        decay_rate = 0.997
        stagnation = 0
        max_stagnation = 60

        # Periodic diversification: every N evaluations, add a random point
        diversify_interval = max(20, budget // 10)
        last_diversify = used_evals

        while used_evals < budget:
            if time.time() - start_time > timeout:
                break

            # Periodic diversification
            if used_evals - last_diversify >= diversify_interval:
                x_rand = rng.random(dim) * span + lo
                x_rand = np.clip(x_rand, lo, hi)
                f_rand = problem(x_rand)
                used_evals += 1
                last_diversify = used_evals

                dominated = False
                for _, f_old in pareto_front:
                    if dominates(f_old, f_rand):
                        dominated = True
                        break
                if not dominated:
                    pareto_front = [(x_p, f_p) for x_p, f_p in pareto_front if not dominates(f_rand, f_p)]
                    pareto_front.append((x_rand, f_rand))
                    stagnation = 0
                    sigma = min(max_sigma, sigma * 1.3)
                continue

            if not pareto_front:
                x = rng.random(dim) * span + lo
                f = problem(x)
                pareto_front.append((x, f))
                used_evals += 1
                continue

            # Select parent using crowding distance (prefer less crowded regions)
            F_arr = np.array([f for _, f in pareto_front])
            cd = crowding_distance(F_arr)
            # Weighted selection: probability proportional to crowding distance
            cd_norm = cd / (cd.sum() + 1e-15)
            parent_idx = rng.choice(len(pareto_front), p=cd_norm)
            parent_x, parent_f = pareto_front[parent_idx]

            # Adaptive mutation: scale sigma based on local density
            # Estimate local density by distance to nearest neighbor
            if len(pareto_front) > 1:
                dists = np.linalg.norm(F_arr - parent_f, axis=1)
                dists[parent_idx] = np.inf
                local_density = np.min(dists)
                # If local density is high, use smaller sigma; if low, larger
                base_sigma = sigma
                if local_density < 0.01:
                    sigma = min(base_sigma, base_sigma * 0.7)
                elif local_density > 0.5:
                    sigma = min(max_sigma, base_sigma * 1.3)
                else:
                    sigma = base_sigma

            # Gaussian mutation
            noise = rng.standard_normal(dim) * sigma * span
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)

            # Evaluate
            child_f = problem(child_x)
            used_evals += 1

            # Update Pareto front
            dominated = False
            for _, f_old in pareto_front:
                if dominates(f_old, child_f):
                    dominated = True
                    break

            if not dominated:
                # Remove dominated points
                pareto_front = [(x_p, f_p) for x_p, f_p in pareto_front if not dominates(child_f, f_p)]
                pareto_front.append((child_x, child_f))
                stagnation = 0
                sigma *= decay_rate
                if sigma < min_sigma:
                    sigma = min_sigma
            else:
                stagnation += 1
                if stagnation > max_stagnation:
                    # Escape local optima: increase sigma and try a random jump
                    sigma = min(max_sigma, sigma * 1.8)
                    stagnation = 0
                    # Also try a random restart point
                    if used_evals < budget:
                        x_jump = rng.random(dim) * span + lo
                        x_jump = np.clip(x_jump, lo, hi)
                        f_jump = problem(x_jump)
                        used_evals += 1
                        dominated_jump = False
                        for _, f_old in pareto_front:
                            if dominates(f_old, f_jump):
                                dominated_jump = True
                                break
                        if not dominated_jump:
                            pareto_front = [(x_p, f_p) for x_p, f_p in pareto_front if not dominates(f_jump, f_p)]
                            pareto_front.append((x_jump, f_jump))
                            stagnation = 0

        return pareto_front
