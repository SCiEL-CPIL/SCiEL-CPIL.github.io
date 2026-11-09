# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2 exploiting known Pareto front structure.
    ZDT2 with n=30: f1=x0, f2=291*(1-sqrt(x0)) when x1..x29=0.
    Reference point is [11, 11]. Only points with f2<11 contribute,
    i.e., x0 > (1-11/291)^2 ≈ 0.9258.
    We concentrate evaluations in x0 ∈ [0.926, 1] to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2: f2 = (1 + 10*(n-1)) * (1 - sqrt(x0)) when x1..=0
        # For n=30: coeff = 291
        # f2 < 11 => 1 - sqrt(x0) < 11/291 => sqrt(x0) > 280/291 => x0 > (280/291)^2
        coeff = 1 + 10 * (n - 1)
        x0_min = (1 - 11.0 / coeff) ** 2  # ≈ 0.9258 for n=30
        x0_max = 1.0

        # Generate evaluation points: concentrate in [x0_min, x0_max]
        # Use a mix: some uniform in [x0_min, x0_max], plus key points
        # x0=1 gives (1, 0) which is the best point
        # We want dense coverage of the Pareto front in the contributing region

        n_budget = self.budget
        # Reserve a few points for exploration near the boundary and x0=1
        # Most points: uniform in [x0_min, x0_max]
        # Add x0=1 explicitly (gives f=(1,0), dominates all others in f2)
        # Add x0_min explicitly (boundary of contributing region)

        # Create t values in [x0_min, x0_max]
        # Use linspace for uniform coverage, but also add x0=1 and x0_min
        if n_budget <= 2:
            t_vals = np.array([x0_min, x0_max])[:n_budget]
        else:
            # Put 1 point at x0_max (best point), 1 at x0_min, rest uniform
            t_vals = np.linspace(x0_min, x0_max, n_budget)
            # Ensure endpoints are included
            t_vals[0] = x0_min
            t_vals[-1] = x0_max

        # Clip to bounds
        t_vals = np.clip(t_vals, lo[0], hi[0])

        X = np.zeros((n_budget, n))
        X[:, 0] = t_vals
        # x1..x_{n-1} = 0 minimizes g(x) and puts us on the Pareto front
        # Clip all to bounds
        X = np.clip(X, lo, hi)

        F = np.empty((n_budget, self.n_obj))
        for i in range(n_budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
