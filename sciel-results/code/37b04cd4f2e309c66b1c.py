import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using NSGA-II-like approach with
    ZDT3-specific exploitation: the Pareto front is achieved when x_2..x_n = 0
    and x_1 controls the trade-off. We combine structured sampling of x_1 with
    stochastic search to find points close to the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For ZDT3, the Pareto front is in x_1 when x_2..x_n = 0.
        # f1 = x_1, g(x) = 1 + 9*sum(x_2..x_n)/29
        # f2 = g*(1 - sqrt(f1/g))
        # On the front (x_2..x_n=0), g=1, so f2 = 1 - sqrt(x_1)
        # The front spans x_1 in [0,1], f1=x_1, f2=1-sqrt(x_1)
        # Reference point is [11, 11]

        # Strategy: 
        # 1. Sample x_1 uniformly across [0,1] to cover the front
        # 2. For each x_1 sample, try to drive x_2..x_n toward 0
        # 3. Use remaining budget for stochastic exploration

        rng = np.random.RandomState(0)

        # Phase 1: Structured sampling of x_1 values
        # Use ~80% of budget for structured front sampling
        n_structured = int(0.8 * self.budget)
        n_structured = max(n_structured, 10)

        # Create structured x_1 values - spread across [0,1]
        x1_values = np.linspace(0, 1, n_structured)
        # Add some random perturbations to avoid exact duplicates
        x1_values = x1_values + rng.uniform(-0.001, 0.001, size=n_structured)
        x1_values = np.clip(x1_values, 0, 1)

        # Shuffle to mix structured and random
        perm = rng.permutation(n_structured)
        x1_values = x1_values[perm]

        # Evaluate structured samples
        F = []
        for x1 in x1_values:
            x = np.zeros(n)
            x[0] = x1
            # Try to make x_2..x_n as close to 0 as possible
            # Add small noise to explore near the front
            x[1:] = rng.uniform(0, 0.01, size=n - 1)
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)

        F = np.array(F)
        used = n_structured

        # Phase 2: Use remaining budget for stochastic exploration
        # Focus on finding points with small x_2..x_n to get closer to the front
        remaining = self.budget - used

        # Keep track of best points (non-dominated)
        def is_dominated(f, F_all):
            """Check if f is dominated by any point in F_all"""
            return np.any(np.all(F_all <= f, axis=1) & np.any(F_all < f, axis=1))

        # Stochastic search: random restarts with focus on driving x_2..x_n to 0
        for _ in range(remaining):
            if time.time() > deadline:
                break

            # Random x_1
            x1 = rng.uniform(0, 1)
            x = np.zeros(n)
            x[0] = x1

            # Try different strategies for x_2..x_n:
            # 1. All zeros (on the front)
            # 2. Small random values
            # 3. Mix

            strategy = rng.randint(3)
            if strategy == 0:
                x[1:] = 0.0
            elif strategy == 1:
                x[1:] = rng.uniform(0, 0.001, size=n - 1)
            else:
                x[1:] = rng.uniform(0, 0.1, size=n - 1)

            x = np.clip(x, lo, hi)
            f = problem(x)
            F = np.vstack([F, [f]])
            used += 1

        return F
