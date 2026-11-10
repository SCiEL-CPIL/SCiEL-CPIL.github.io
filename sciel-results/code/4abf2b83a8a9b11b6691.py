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

        # ZDT6 analytic structure:
        # f1(x) = 1 - exp(-4*x0) * sin(10*pi*x0)
        # f2(x) = g(x)^(1 + 1/(2*f1+4))
        # g(x) = 1 + 9/(n-1) * sum(x_i, i=1..n-1)
        # Pareto front: x1..x9 = 0, g=1, f2 = 1/(f1+4)^(1/(2*f1+4))... wait
        # Actually f2 = g^(1 + 1/(2*f1+4)), with g=1: f2 = 1
        # No, let me reconsider. The Pareto front of ZDT6:
        # f1 = 1 - exp(-4*x0)*sin(10*pi*x0)
        # f2 = (1 + 9/(n-1)*sum(x1..x_{n-1}))^(1 + 1/(2*f1+4))
        # On Pareto front, x1..x_{n-1} = 0, so g = 1
        # f2 = 1^(anything) = 1? That's not right.
        # Actually for ZDT6, the Pareto front is: f2 = 1/(f1+4)^(1/(2*f1+4))... 
        # Let me just use the problem to evaluate.

        # Strategy: 
        # 1. Use known structure to initialize in promising regions
        # 2. The front is dense near f1=0, so sample x0 values that give f1 near 0
        # 3. Keep x1..x9 near 0 for good g values
        # 4. Use local refinement

        # For f1 near 0: 1 - exp(-4*x0)*sin(10*pi*x0) = 0
        # sin(10*pi*x0) = exp(-4*x0)
        # Near x0=0: sin(10*pi*x0) ≈ 10*pi*x0, exp(-4*x0) ≈ 1
        # So 10*pi*x0 ≈ 1, x0 ≈ 1/(10*pi) ≈ 0.0318
        # Also x0=0 gives f1=0
        # x0=0.1: sin(pi)=0, f1=1
        # x0=0.2: sin(2*pi)=0, f1=1
        # The function oscillates. For f1 in [0,1], we need various x0 values.

        # Better approach: sample x0 to cover the range of f1 values
        # f1(x0) = 1 - exp(-4*x0)*sin(10*pi*x0)
        # This oscillates between 0 and 1 (approximately) as x0 goes from 0 to 1

        # Initialize with diverse x0 values, x1..x9 near 0
        # Use a mix of:
        # - x0 values that give f1 near 0 (dense region)
        # - x0 values that give f1 spread across [0,1]

        # Let's create initial population targeting different f1 regions
        # The hypervolume is maximized by good coverage, especially in dense regions

        # Analytic initialization: pick x0 values, set x1..x9=0
        # Then refine locally

        # Generate candidate x0 values
        # f1 oscillates with period 0.2 in x0
        # For good coverage, sample x0 in [0, 1] but weighted toward regions with low f1

        # Let's use a combination:
        # 1. Uniform in x0 to get spread
        # 2. Target specific f1 values by solving for x0

        # Simpler: just use good initial points and local search
        # Key insight: for ZDT6, x1..x9 should be near 0 for Pareto optimality
        # So we can reduce effective dimensionality

        # Initialize: 20 points with diverse x0, x1..x9 = 0
        n_init = 20
        # x0 values: spread across [0,1] but with extra near 0
        x0_vals = np.concatenate([
            np.linspace(0, 0.1, 10),  # Dense near 0 where front is dense
            np.linspace(0.1, 1.0, 10)  # Spread across rest
        ])

        X = np.zeros((n_init, dim))
        X[:, 0] = x0_vals
        # x1..x9 = 0 for initial (Pareto optimal direction)

        F = np.array([problem(x) for x in X])
        used = n_init

        # Local refinement: for each point, do a few local steps
        # Use coordinate descent or small random perturbations

        # Time check
        if time.time() > deadline - 50:
            return X

        # Local search: perturb x0 and x1..x9 slightly
        # Focus on improving hypervolume by moving points to better positions

        # Simple local search: for each point, try small perturbations
        step_size = 0.01
        n_local = min(10, (budget - used) // n_init)

        for iteration in range(n_local):
            if time.time() > deadline - 30:
                break
            for i in range(n_init):
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                # Try perturbing x0
                x0_new = X[i, 0] + step_size * (np.random.random() * 2 - 1)
                x0_new = np.clip(x0_new, lo[0], hi[0])
                x_new = X[i].copy()
                x_new[0] = x0_new
                f_new = problem(x_new)
                used += 1

                # Accept if it improves (non-dominated or better in some way)
                # Simple: accept if it's not dominated by current population
                # For speed, just accept if sum of objectives is lower
                if np.sum(f_new) < np.sum(F[i]):
                    X[i] = x_new
                    F[i] = f_new

                # Also try perturbing one of x1..x9 (should stay near 0)
                if used < budget:
                    j = np.random.randint(1, dim)
                    xj_new = X[i, j] + step_size * 0.5 * (np.random.random() * 2 - 1)
                    xj_new = np.clip(xj_new, lo[j], hi[j])
                    x_new2 = X[i].copy()
                    x_new2[j] = xj_new
                    f_new2 = problem(x_new2)
                    used += 1
                    if np.sum(f_new2) < np.sum(F[i]):
                        X[i] = x_new2
                        F[i] = f_new2

        # Final: if budget remains, add more diverse points
        while used < budget:
            if time.time() > deadline - 5:
                break
            # Random point with x1..x9 biased toward 0
            x = np.random.random(dim)
            x[1:] *= 0.1  # Bias toward 0 for x1..x9
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1

            # Replace worst point if this is better
            worst_idx = np.argmax(np.sum(F, axis=1))
            if np.sum(f) < np.sum(F[worst_idx]):
                X[worst_idx] = x
                F[worst_idx] = f

        return X
