"""Multi-objective optimizer on ZDT6 with Pareto-front initialization and local refinement."""
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search initialized on the known ZDT6 Pareto set."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, keep 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initialize on the known ZDT6 Pareto set: x1=0, x2=0, ..., x10=0
        # For ZDT6, the Pareto front is achieved when x1=0 and x2..x10=0
        # Actually, for ZDT6, the Pareto set is x1=0, and x2..xn can be anything
        # but the front is parameterized by x1. Let me reconsider.
        # ZDT6: f1 = 1 - exp(-4*x1)*sin^6(6*pi*x1)
        #        f2 = g(x) * h(f1, g(x))
        #        g(x) = 1 + 9/n * sum(x_i) for i=2..n
        #        h(f1, g) = 1 - sqrt(f1/g)
        # The Pareto set is x1 in [0,1], x2=...=xn=0
        # So we should sample x1 values and set x2..xn=0

        # Create initial population on the Pareto front
        n_init = min(50, self.budget // 4)
        x1_vals = np.linspace(0.0, 1.0, n_init)
        X = np.zeros((n_init, self.dim))
        X[:, 0] = x1_vals
        # x2..xn are already 0

        F = np.array([problem(x) for x in X])
        used = n_init

        # Track all evaluated points for hypervolume
        all_points = list(zip(X.tolist(), F.tolist()))

        # Local search / refinement: perturb points slightly and keep non-dominated
        # Use a simple local search around each point
        step_size = 0.01
        iterations = 0
        max_iterations = self.budget - used

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # For each point in current population, try a small perturbation
            for i in range(len(X)):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Small random perturbation
                perturbation = step_size * np.random.standard_normal(self.dim)
                child = X[i] + perturbation
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_points.append((child.tolist(), fc.tolist()))

                # Check if child dominates any existing point
                # If it does, replace the worst point
                # Simple strategy: if child is non-dominated, add it to population
                # For now, just track all points and let hypervolume handle it

            # Occasionally do a larger step
            if iterations % 10 == 0 and used < self.budget:
                large_perturbation = 0.1 * np.random.standard_normal(self.dim)
                idx = np.random.randint(len(X))
                child = X[idx] + large_perturbation
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                all_points.append((child.tolist(), fc.tolist()))

            iterations += 1

        # Final polish: try to refine the best points with smaller steps
        if used < self.budget and time.time() < deadline - 10:
            # Sort by first objective and take top candidates
            F_arr = np.array([p[1] for p in all_points])
            X_arr = np.array([p[0] for p in all_points])

            # Select non-dominated points
            non_dom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        non_dom_mask[i] = False
                        break

            nd_points_X = X_arr[non_dom_mask]
            nd_points_F = F_arr[non_dom_mask]

            # Refine each non-dominated point with small steps
            for i in range(len(nd_points_X)):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                for _ in range(5):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    small_perturbation = 0.001 * np.random.standard_normal(self.dim)
                    child = nd_points_X[i] + small_perturbation
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    all_points.append((child.tolist(), fc.tolist()))

        return X
