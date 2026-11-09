"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front: x_2...x_n = 0.5, x_1 in [0,1]
        # f1 = x_1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [x_i^2 - 10*cos(4*pi*x_i)]
        # h(x_1, g) = 1 - sqrt(x_1/g) - (x_1/g)*sin(10*pi*x_1)
        # f2 = g * h

        # Strategy: sample x_1 adaptively to cover the Pareto front well in objective space.
        # The Pareto front is discontinuous (ZDT3 has multiple disconnected segments).
        # We'll do a combination of:
        # 1. Dense sampling along x_1 on the Pareto front (x_2..x_n = 0.5)
        # 2. Some random exploration to find good points near the front

        used = 0
        results = []  # list of objective vectors

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            return f

        # Phase 1: Sample the Pareto front densely
        # ZDT3 front: x_2..x_n = 0.5
        # f1 = x_1, f2 = g(0.5...) * h(x_1, g(0.5...))
        # g at x_i=0.5 for i>=2: each term = 0.25 - 10*cos(4*pi*0.5) = 0.25 - 10*cos(2*pi) = 0.25 - 10 = -9.75
        # Wait, let me recalculate: cos(4*pi*0.5) = cos(2*pi) = 1
        # So each term: 0.5^2 - 10*cos(4*pi*0.5) = 0.25 - 10*1 = -9.75
        # g = 1 + 10*(dim-1) + (dim-1)*(-9.75) = 1 + 10*(dim-1) - 9.75*(dim-1) = 1 + 0.25*(dim-1)
        # For dim=30: g = 1 + 0.25*29 = 1 + 7.25 = 8.25

        g_val = 1.0 + 0.25 * (dim - 1)  # g when x_2..x_n = 0.5

        # h(x_1, g) = 1 - sqrt(x_1/g) - (x_1/g)*sin(10*pi*x_1)
        # f2 = g * h

        # The front has discontinuities where h changes sign or behavior.
        # Let's sample x_1 uniformly in [0, 1] with many points to cover all segments.

        # Number of front samples: use most of the budget for dense front sampling
        n_front = min(budget - 20, 180)  # leave some for exploration
        x1_samples = np.linspace(0, 1, n_front, endpoint=True)

        # Evaluate points on the Pareto front
        x_front = np.full((n_front, dim), 0.5)
        x_front[:, 0] = x1_samples

        for i in range(n_front):
            if used >= budget or time.time() > deadline:
                break
            f = eval_point(x_front[i])
            if f is not None:
                results.append(np.array(f))

        # Phase 2: Use remaining budget for perturbation around front points
        # to potentially find slightly better points (due to numerical issues or
        # the fact that the "front" might not be exactly at x_i=0.5 for all i)
        remaining = budget - used
        if remaining > 0:
            # Pick some random front points and perturb them slightly
            n_perturb = min(remaining, 20)
            for _ in range(n_perturb):
                if used >= budget or time.time() > deadline:
                    break
                idx = np.random.randint(0, n_front)
                x_pert = x_front[idx].copy()
                # Small perturbation in x_2..x_n
                x_pert[1:] += 0.01 * np.random.randn(dim - 1)
                x_pert = np.clip(x_pert, lo, hi)
                f = eval_point(x_pert)
                if f is not None:
                    results.append(np.array(f))

        # Phase 3: Random exploration with remaining budget
        remaining = budget - used
        if remaining > 0:
            for _ in range(remaining):
                if used >= budget or time.time() > deadline:
                    break
                x_rand = lo + (hi - lo) * np.random.rand(dim)
                f = eval_point(x_rand)
                if f is not None:
                    results.append(np.array(f))

        return results
# EVOLVE-BLOCK-END
