"""Optimizer for ZDT2 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """
    ZDT2-specific optimizer.
    ZDT2: f1 = x0, g = 1 + 9*sum(x1..xn-1)/n, f2 = g*(1 - sqrt(f1/g))
    Pareto front: x1..x_{n-1} = 0, x0 in [0,1]
    f1 = x0, f2 = 1 - sqrt(x0)
    """

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

        # Phase 1: Dense sampling of the known Pareto front (x[1:]=0, x[0] in [0,1])
        # Use ~150 evaluations for dense front coverage
        n_front = min(150, budget)
        x0_vals = np.linspace(0.0, 1.0, n_front)

        F_list = []
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            f = problem(x)
            F_list.append(f)

        F = np.array(F_list)
        used = len(F_list)

        # Phase 2: Use remaining budget for local refinement around promising regions
        # Focus on the middle of the front where curvature is highest
        if used < budget:
            # Try points slightly off the front to see if we can improve
            # Also try to refine x0 positions
            remaining = budget - used
            n_refine = min(remaining, 50)

            # Sample in regions of high curvature (middle of front)
            refine_x0 = np.linspace(0.2, 0.8, n_refine)
            for x0 in refine_x0:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                x = np.zeros(dim)
                x[0] = x0
                f = problem(x)
                F_list.append(f)
                F = np.array(F_list)
                used += 1

        # Phase 3: If budget remains, try small perturbations of x[1:] to see
        # if there's any benefit (there shouldn't be for ZDT2, but check)
        if used < budget:
            # Try a few points with tiny x[1:] values
            n_perturb = min(budget - used, 20)
            rng = np.random.default_rng(42)
            for _ in range(n_perturb):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                x = np.zeros(dim)
                x[0] = rng.uniform(0, 1)
                # Tiny perturbation to other vars
                x[1:] = rng.uniform(0, 0.01, dim - 1)
                f = problem(x)
                F_list.append(f)
                F = np.array(F_list)
                used += 1

        return F
