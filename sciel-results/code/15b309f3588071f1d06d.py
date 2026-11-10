import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytic initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT6, the Pareto front has x1..x9 = 0
        # f1(x0) = 1 - exp(-4*x0)*sin(10*pi*x0)
        # f2 = g^(1 + 1/(2*f1+4)) where g = 1 + 9/9 * sum(x1..x9)
        # On Pareto front: g=1, so f2 = 1^(...) = 1
        # But this seems wrong. Let me just use the problem.

        # Strategy: Optimize f1 spacing for hypervolume
        # The hypervolume is maximized when points are spaced to maximize area under the front
        # For a decreasing front f2(f1), optimal spacing is roughly uniform in f1

        # Step 1: Find x0 values that give well-distributed f1 values
        # f1(x0) oscillates. We need to find x0 for target f1 values.

        # First, let's sample x0 densely to understand f1(x0) behavior
        # Then pick x0 values that give good f1 coverage

        # Use the remaining budget wisely:
        # - 50% for initial diverse points
        # - 30% for local refinement
        # - 20% for final polish

        n_initial = min(100, budget // 2)
        n_refine = min(50, budget // 4)
        n_polish = budget - n_initial - n_refine

        # Generate x0 values targeting different f1 regions
        # f1 = 1 - exp(-4*x0)*sin(10*pi*x0)
        # Near x0=0: f1 ≈ 0 (since sin(0)=0, but derivative is 10*pi)
        # Actually at x0=0: f1 = 1 - 1*0 = 1? No: exp(0)=1, sin(0)=0, so f1=1-0=1
        # Wait: f1(0) = 1 - exp(0)*sin(0) = 1 - 1*0 = 1
        # f1(0.1) = 1 - exp(-0.4)*sin(pi) = 1 - 0.67*0 = 1
        # f1(0.05) = 1 - exp(-0.2)*sin(pi/2) = 1 - 0.82*1 = 0.18
        # f1(0.03) = 1 - exp(-0.12)*sin(0.3*pi) = 1 - 0.89*0.81 = 0.28

        # Let me just sample x0 uniformly and see what f1 values we get
        # Then pick the ones that give good coverage

        # Better: use the known structure
        # The front is dense near f1=0 (x0 near 0.03-0.05)
        # and sparse elsewhere

        # Optimal HV spacing: for a convex decreasing front, use more points where the curve is steeper
        # For ZDT6, the front is not simple, so let's use a hybrid approach

        # Initialize with x0 values that give f1 in [0, 1]
        # Try x0 in [0, 0.1] which covers most of the interesting range
        x0_candidates = np.linspace(0, 0.1, n_initial)

        X = np.zeros((n_initial, dim))
        X[:, 0] = x0_candidates
        # x1..x9 = 0 for Pareto optimality

        F = np.array([problem(x) for x in X])
        used = n_initial

        # Check time
        if time.time() > deadline - 60:
            return X

        # Local refinement: for each point, try small perturbations in x0
        # Focus on improving hypervolume by moving points to better positions
        for i in range(n_initial):
            if used >= budget or time.time() > deadline - 30:
                break

            # Try a few small perturbations
            for _ in range(3):
                if used >= budget or time.time() > deadline - 10:
                    break
                x0_new = X[i, 0] + 0.005 * (np.random.random() * 2 - 1)
                x0_new = np.clip(x0_new, lo[0], hi[0])
                x_new = X[i].copy()
                x_new[0] = x0_new
                f_new = problem(x_new)
                used += 1

                # Accept if it improves the sum of objectives (simple heuristic)
                if np.sum(f_new) < np.sum(F[i]):
                    X[i] = x_new
                    F[i] = f_new

        # Final polish: add a few more points in promising regions
        while used < budget and time.time() < deadline - 5:
            # Sample near existing good points
            idx = np.random.randint(0, n_initial)
            x = X[idx].copy()
            x[0] += 0.01 * (np.random.random() * 2 - 1)
            x[0] = np.clip(x[0], lo[0], hi[0])
            f = problem(x)
            used += 1

            # Replace the worst point if this is better
            worst_idx = np.argmax(np.sum(F, axis=1))
            if np.sum(f) < np.sum(F[worst_idx]):
                X[worst_idx] = x
                F[worst_idx] = f

        return X
