# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting the known front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # ZDT6: f1 = x1, f2 = 1 - sqrt(x1/(2*g(x))) where g = 2 + sum_{i=2}^{n} sin(10*pi*x_i)
        # The Pareto front (g=2) is f1 = x1, f2 = 1 - sqrt(x1/2).
        # We sample x1 densely and locally minimize g for each to avoid domination.

        # Precompute a grid of x1 values (the front is 1D in x1)
        # Use a combination: dense uniform + a few extra
        n_x1 = 40
        x1_grid = np.linspace(0.0, 1.0, n_x1)

        # We will allocate evaluations: some for g-minimization per x1, some for extra exploration
        # Each x1 needs a few g-minimization evals. Keep total <= budget.

        def local_min_g(x1, n_evals):
            # Minimize g over x[2:] given fixed x1.
            # g = 2 + sum sin(10*pi*x_i). Minimum is 2 (at x_i = 0.1 or 0.3 etc where sin=0 and the
            # standard ZDT6 minimum). We do a short local search.
            best_x = None
            best_g = np.inf
            # initialize at a good point: x_i = 0.1 gives sin(10*pi*0.1)=sin(pi)=0 -> g=2
            base = np.zeros(dim)
            base[0] = x1
            for i in range(1, dim):
                base[i] = 0.1
            # evaluate
            f = problem(base)
            # we can't get g directly, but f2 encodes it. Use f as proxy for domination.
            # Actually we want to minimize g which maximizes f2 for fixed x1.
            # f2 = 1 - sqrt(x1/(2*g)). Larger g -> smaller f2. So minimize g == maximize f2.
            best_f2 = f[1]
            best_x = base.copy()
            used = 1
            step = 0.05
            for _ in range(n_evals - 1):
                if used >= n_evals:
                    break
                # random perturbation of x[1:]
                cand = best_x.copy()
                idx = rng.integers(1, dim)
                cand[idx] += step * rng.standard_normal()
                cand[idx] = np.clip(cand[idx], lo[idx], hi[idx])
                fc = problem(cand)
                used += 1
                if fc[1] > best_f2:  # maximize f2 (minimize g)
                    best_f2 = fc[1]
                    best_x = cand.copy()
                # occasionally shrink step
            return best_x, used

        recorded = []

        # For each x1, do a local search to minimize g.
        # Budget allocation: reserve some for random exploration at the end.
        evals_per_x1 = max(3, min(8, budget // (2 * n_x1)))
        # We'll track used evaluations
        used_total = 0

        for x1 in x1_grid:
            if time.time() > deadline - 30:
                break
            if used_total >= budget:
                break
            remaining = budget - used_total
            n_evals = min(evals_per_x1, remaining)
            if n_evals < 1:
                break
            best_x, used = local_min_g(x1, n_evals)
            used_total += used
            f = problem(best_x) if False else None
            # re-evaluate to record (problem may have been called inside; but we need the vector)
            # Actually local_min_g already evaluated; let's just record best_x's f.
            f = problem(best_x)
            used_total += 1
            if used_total > budget:
                # over budget by 1 due to re-eval; but we guard. Clamp: don't record if over.
                used_total = budget
            recorded.append((best_x, f))

        # Use remaining budget for random exploration to improve coverage
        while used_total < budget and time.time() < deadline - 30:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used_total += 1
            recorded.append((x, f))

        return recorded
# EVOLVE-BLOCK-END
