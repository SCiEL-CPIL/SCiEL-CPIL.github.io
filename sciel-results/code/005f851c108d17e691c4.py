"""Multi-objective optimizer for WFG4 using deterministic quasi-random sampling + local refinement."""
import numpy as np
import time


class Optimizer:
    """Deterministic quasi-random (Sobol-like via Halton) sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Generate Halton sequence points
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            """Generate n points in [0,1)^dim using Halton sequence."""
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            for d in range(dim):
                base = primes[d % len(primes)]
                for i in range(n):
                    points[i, d] = halton(i + 1, base)
            return points

        # Phase 1: Use most of the budget for systematic quasi-random sampling
        # Reserve some evaluations for local refinement
        n_sample = int(budget * 0.85)
        n_refine = budget - n_sample

        # Generate quasi-random points
        Q = halton_sequence(n_sample, dim)
        # Map to bounds
        X = lo + Q * span

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sample

        # Check time
        if time.time() > deadline - 30.0:
            return X

        # Phase 2: Local refinement around best non-dominated points
        # Find non-dominated points
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fi in F:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False

        # Get non-dominated set
        nd_indices = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)

        if len(nd_indices) == 0:
            nd_indices = [0]

        # Refine around the best non-dominated points
        # Use a local search with small perturbations
        n_parents = min(10, len(nd_indices))
        parents = [nd_indices[i] for i in range(n_parents)]

        # For each parent, try a few local perturbations
        attempts_per_parent = max(1, n_refine // len(parents))
        for p_idx in parents:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            base_x = X[p_idx]
            base_f = F[p_idx]

            for attempt in range(attempts_per_parent):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Create a perturbation with decreasing step size
                step_size = 0.05 * (1.0 - attempt / max(attempts_per_parent, 1))
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                child_x = base_x + step_size * span * direction
                child_x = np.clip(child_x, lo, hi)

                child_f = problem(child_x)
                used += 1

                # Replace the worst point in the population
                # Use a simple crowding-like criterion: replace point with largest sum
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child_x
                F[worst_idx] = child_f

        return X
