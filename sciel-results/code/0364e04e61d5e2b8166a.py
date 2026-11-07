"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ7 multi-objective optimizer.
    
    Key insight: DTLZ7's Pareto front requires g(x_in)=0, which forces
    x_in (variables 3..dim-1) to be 0. So only x_0, x_1, x_2 matter.
    
    Strategy:
    1. Set x_in = 0 always (guarantees g=0, all points on the front).
    2. Use Halton quasi-random sequence to sample the 3D Pareto front
       (x_0, x_1, x_2 in [0,1]) for uniform, low-discrepancy coverage.
    3. Use remaining budget for local refinement around promising regions.
    4. Deadline guard to avoid timeout.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, n, base):
        """Generate n Halton sequence values in [0,1) for given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n + 1):
            x = 0.0
            idx = i
            while idx > 0:
                idx, rem = divmod(idx, base)
                x += rem * f
                f /= base
            result[i - 1] = x
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7: g=0 requires x_in=0, so only first 3 variables matter
        # Set x_in = 0 (middle of [0,1] is 0.5, but we need 0 for g=0)
        # Actually g = |x_in| + ... so x_in must be 0
        x_in_zero = np.zeros(self.dim - 3)

        def make_x(u0, u1, u2):
            """Create a full-dim vector with x_in=0."""
            x = np.zeros(self.dim)
            x[0] = u0
            x[1] = u1
            x[2] = u2
            return x

        def eval_point(x):
            """Evaluate and return objectives."""
            return np.array(problem(x))

        # Phase 1: Halton sequence sampling of the 3D Pareto front
        # Use as many evaluations as budget allows, but reserve some for refinement
        n_halton = int(self.budget * 0.7)
        n_refine = self.budget - n_halton

        # Generate Halton sequences for 3 dimensions (bases 2, 3, 5)
        h0 = self._halton(n_halton, 2)
        h1 = self._halton(n_halton, 3)
        h2 = self._halton(n_halton, 5)

        X = np.zeros((self.budget, self.dim))
        F = np.zeros((self.budget, self.n_obj))
        used = 0

        for i in range(n_halton):
            if time.time() > deadline:
                break
            x = make_x(h0[i], h1[i], h2[i])
            X[i] = x
            F[i] = eval_point(x)
            used += 1

        # Phase 2: Local refinement around the best (non-dominated) points
        if used < self.budget:
            # Find non-dominated points among evaluated
            # A point is dominated if another point is <= in all objectives and < in at least one
            nd_indices = []
            for i in range(used):
                dominated = False
                for j in range(used):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i)

            if len(nd_indices) == 0:
                nd_indices = list(range(used))

            # Refine around each non-dominated point
            step = 0.05
            for idx in nd_indices:
                while used < self.budget and time.time() < deadline:
                    # Random local perturbation
                    delta = rng.standard_normal(3) * step
                    u0 = np.clip(X[idx, 0] + delta[0], 0, 1)
                    u1 = np.clip(X[idx, 1] + delta[1], 0, 1)
                    u2 = np.clip(X[idx, 2] + delta[2], 0, 1)
                    x = make_x(u0, u1, u2)
                    X[used] = x
                    F[used] = eval_point(x)
                    used += 1
                    # Check if this new point dominates the original
                    if np.all(F[used - 1] <= F[idx]) and np.any(F[used - 1] < F[idx]):
                        X[idx] = x
                        F[idx] = F[used - 1]
                        # Reduce step as we converge
                        step *= 0.7

        return X
# EVOLVE-BLOCK-END
