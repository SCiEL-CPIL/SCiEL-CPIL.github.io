"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using a targeted analytical approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # ZDT6 specific analytical approach
        # f1(x) = 1 - exp(-4 * sum(x_i * sin(10 * pi * x_i + 1))) + 0.1 * t * (1 + 10*(m-1))
        #       where t = sum((x_i - 0.5)^2 / (4*x_i*(1-x_i) + 1e-20) - 0.5*sin(2*pi*(x_i+0.5)))
        # f2(x) = g(x) * h(g(x), f1(x))
        #       where g(x) = 1 + 10*(m-1) * (sum(x_i)/n)^alpha
        #       h(g, f) = 1 - sqrt(f/g)
        #       alpha = 0.5 + 2 * g(x) * (x_1^2 + x_1) / (sum(x_i^2 + x_i) + 1e-20)

        # Key insight: To minimize f2, we need small g(x) and f1 close to g.
        # g(x) is minimized when sum(x_i)/n is small, i.e., x_i close to 0.
        # But f1 has a complex structure. The Pareto front of ZDT6 is known to be
        # non-convex and discontinuous.

        # Strategy: 
        # 1. Generate diverse initial points
        # 2. Use a local search that exploits the structure of ZDT6
        # 3. Maintain an archive of non-dominated points

        archive_X = []
        archive_F = []
        used = 0

        def add_point(x, f):
            nonlocal used
            if time.time() > deadline:
                return
            # Check if f is dominated by any point in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, don't add
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            archive_X.append(x)
            archive_F.append(f)

        # Phase 1: Analytical exploration of the Pareto front
        # The ZDT6 Pareto front can be parameterized. Let's sample along it.
        # For ZDT6, the optimal solutions have specific structure.
        # Let's try points where x_1 varies and others are set to minimize g.

        # Generate initial diverse population
        init_pop = 40
        if self.budget >= init_pop:
            for i in range(init_pop):
                if time.time() > deadline:
                    break
                if i < 10:
                    # Points near origin (small g)
                    x = np.zeros(n)
                    x[0] = i / 10.0 * 0.5
                    x[1:] = 0.01
                elif i < 20:
                    # Points with moderate values
                    x = rng.random(n) * 0.5
                    x[0] = (i - 10) / 10.0
                else:
                    # Random points
                    x = lo + (hi - lo) * rng.random(n)
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                add_point(x, f)

        # Phase 2: Targeted local search
        # For each archive point, try to improve by adjusting variables
        remaining = self.budget - used
        if remaining > 0:
            # Sort archive by f1 to focus on different regions
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            order = np.argsort(archive_F_arr[:, 0])
            
            # Use remaining budget for local search
            n_local = min(len(archive_X_arr), max(1, remaining // 20))
            for idx in order[:n_local]:
                if used >= self.budget or time.time() > deadline:
                    break
                base_x = archive_X_arr[idx].copy()
                base_f = archive_F_arr[idx].copy()
                
                # Try small perturbations
                for step in [0.01, 0.05, 0.1]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    for dim_idx in range(n):
                        if used >= self.budget or time.time() > deadline:
                            break
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x = base_x.copy()
                            x[dim_idx] += direction * step
                            x = np.clip(x, lo, hi)
                            f = problem(x)
                            used += 1
                            if np.all(f <= base_f) and np.any(f < base_f):
                                base_x = x
                                base_f = f
                                add_point(x, f)
                            elif not np.any(np.all(np.array(archive_F) <= f, axis=1) & np.any(np.array(archive_F) < f, axis=1)):
                                add_point(x, f)

        # Phase 3: If budget remains, do more random search with exploitation
        if used < self.budget:
            while used < self.budget:
                if time.time() > deadline:
                    break
                if archive_X:
                    # Pick a random archive point and mutate
                    idx = rng.integers(len(archive_X))
                    base = archive_X[idx].copy()
                    # Adaptive mutation
                    sigma = 0.05 + 0.1 * rng.random()
                    x = base + sigma * (hi - lo) * rng.standard_normal(n)
                    x = np.clip(x, lo, hi)
                else:
                    x = lo + (hi - lo) * rng.random(n)
                f = problem(x)
                used += 1
                add_point(x, f)

        return archive_X
# EVOLVE-BLOCK-END
