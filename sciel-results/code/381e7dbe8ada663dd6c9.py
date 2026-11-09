"""Rewritten optimizer: Multi-start local search with diverse initializations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with diverse initializations for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            calls_used += 1
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Generate diverse initial points
        # Strategy: mix of random, structured grid-like points, and origin
        initial_points = []

        # Origin and nearby points
        initial_points.append(np.zeros(dim))
        initial_points.append(np.ones(dim) * 0.5)
        initial_points.append(-np.ones(dim) * 0.5)
        initial_points.append(np.ones(dim) * 2.0)
        initial_points.append(-np.ones(dim) * 2.0)

        # Structured points in the center region
        for i in range(-2, 3):
            p = np.zeros(dim)
            p[0] = i
            initial_points.append(p)
            p2 = np.zeros(dim)
            p2[1] = i
            initial_points.append(p2)

        # Random points in [-4, 4]^dim
        for _ in range(20):
            initial_points.append(rng.uniform(-4, 4, dim))

        # Sort by distance from origin (prefer closer points first for early progress)
        initial_points.sort(key=lambda p: np.linalg.norm(p))

        # Allocate budget: use most of budget on the best few starts
        # Try each start with a limited number of Nelder-Mead iterations
        points_per_start = max(1, budget // (len(initial_points) + 10))

        for point in initial_points:
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Nelder-Mead style local search
            # Create simplex around this point
            step = 0.5
            simplex = [point.copy()]
            for d in range(dim):
                v = point.copy()
                v[d] += step
                simplex.append(v)
                v = point.copy()
                v[d] -= step
                simplex.append(v)

            # Evaluate simplex
            simplex_f = []
            for v in simplex:
                simplex_f.append(counted_func(v))
                if time.time() > deadline or calls_used >= budget:
                    break

            if len(simplex_f) < len(simplex):
                break

            # Nelder-Mead iterations
            max_nm_iter = min(30, (budget - calls_used) // (dim + 1))
            for _ in range(max_nm_iter):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Sort simplex by fitness
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                reflected = centroid + (centroid - worst)
                ref_f = counted_func(reflected)
                if time.time() > deadline or calls_used >= budget:
                    break

                # Check if we can stop
                if ref_f < simplex_f[0]:
                    # Expansion
                    expanded = centroid + 2 * (centroid - worst)
                    exp_f = counted_func(expanded)
                    if time.time() > deadline or calls_used >= budget:
                        break
                    if exp_f < ref_f:
                        simplex[-1] = expanded
                        simplex_f[-1] = exp_f
                    else:
                        simplex[-1] = reflected
                        simplex_f[-1] = ref_f
                elif ref_f < simplex_f[-2]:
                    simplex[-1] = reflected
                    simplex_f[-1] = ref_f
                else:
                    # Contraction
                    contracted = centroid + 0.5 * (worst - centroid)
                    con_f = counted_func(contracted)
                    if time.time() > deadline or calls_used >= budget:
                        break
                    if con_f < worst_f:
                        simplex[-1] = contracted
                        simplex_f[-1] = con_f
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])
                            if time.time() > deadline or calls_used >= budget:
                                break

            # After Nelder-Mead, do a few random restarts around best found so far
            if calls_used < budget and time.time() < deadline:
                # Local random search around current best
                for _ in range(min(10, (budget - calls_used) // 2)):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    perturbation = rng.normal(0, 0.3, dim)
                    candidate = best_x + perturbation
                    counted_func(candidate)

        return best_x, best_f
# EVOLVE-BLOCK-END
