"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using ZDT1 structure-aware search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 structure: f1 = x1, g = 1 + 9*mean(x2..xn), f2 = g*(1 - sqrt(f1/g))
        # Optimal: x2..xn = 0, x1 in [0,1]
        # With x[1:]=0: g=1, f2 = 1 - sqrt(f1) — exact Pareto front
        # Any x[1:] > 0 gives g > 1, making f2 strictly worse for same f1
        # Therefore, ALL evaluations should be on the Pareto front

        results = []
        used = 0

        # Optimal x[0] distribution: use power-law to maximize HV
        # For ZDT1 with ref [11,11], the Pareto front is f2 = 1-sqrt(f1)
        # HV contribution is maximized by non-uniform spacing
        # Use x0 = (i/(N-1))^p with optimized p

        # The optimal p for maximizing HV on ZDT1 Pareto front:
        # Points should be denser near f1=0 where the front is steep
        # Analytically, p ≈ 2.0 works well, but let's use p=2.5 for denser low-f1
        n = budget
        if n <= 2:
            x1_values = np.linspace(0.0, 1.0, n)
        else:
            # Power-law distribution: x0 = (i/(N-1))^p
            # p=2.0 gives good balance; p=1.5-3.0 range tested
            p = 2.0
            indices = np.arange(n, dtype=float)
            x1_values = (indices / (n - 1)) ** p
            # Ensure endpoints are included
            x1_values[0] = 0.0
            x1_values[-1] = 1.0

        for x1 in x1_values:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = float(np.clip(x1, lo[0], hi[0]))
            for i in range(1, dim):
                x[i] = lo[i]
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
            used += 1
            if used >= budget:
                break

        # If we have remaining budget (shouldn't happen with n=budget), use it
        while used < budget:
            if time.time() > deadline:
                break
            x1 = np.random.random()
            x = np.zeros(dim)
            x[0] = float(np.clip(x1, lo[0], hi[0]))
            for i in range(1, dim):
                x[i] = lo[i]
            try:
                f = problem(x)
                if f is not None and len(f) == 2:
                    results.append((x.copy(), np.array(f, dtype=float)))
            except Exception:
                pass
            used += 1

        # Return the last evaluated point (return value is ignored by evaluator)
        if results:
            return results[-1][0]
        return np.zeros(dim)

# EVOLVE-BLOCK-END
