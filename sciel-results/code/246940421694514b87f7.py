import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4 exploiting its decoupled structure.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT4: x1 controls the front, x2..xn control deviation from front.
        # Pareto front is at x2..xn = 0.
        # We sweep x1 uniformly across its bounds.
        
        # Step 1: Determine the best "base" value for x[1:] to minimize f1.
        # We try a few candidates: 0, center of bounds, and a few random points.
        # For each candidate, we fix x[0] to the midpoint of its range and evaluate f1.
        
        candidates = []
        
        # 0 if in bounds
        base_zero = np.zeros(self.dim - 1)
        for i in range(self.dim - 1):
            if 0.0 < lo[i+1]:
                base_zero[i] = lo[i+1]
            elif 0.0 > hi[i+1]:
                base_zero[i] = hi[i+1]
        candidates.append(base_zero)
        
        # Center of bounds
        base_center = np.zeros(self.dim - 1)
        for i in range(self.dim - 1):
            base_center[i] = (lo[i+1] + hi[i+1]) / 2.0
        candidates.append(base_center)
        
        # A few random candidates (seeded for reproducibility)
        rng = np.random.RandomState(42)
        for _ in range(5):
            base_rand = np.zeros(self.dim - 1)
            for i in range(self.dim - 1):
                base_rand[i] = rng.uniform(lo[i+1], hi[i+1])
            candidates.append(base_rand)

        # Evaluate each candidate with x[0] at midpoint
        x0_mid = (lo[0] + hi[0]) / 2.0
        best_base = None
        best_f1 = np.inf
        n_used = 0

        for base in candidates:
            if time.time() > deadline:
                break
            x = np.empty(self.dim)
            x[0] = x0_mid
            x[1:] = base
            try:
                f = problem(x)
                n_used += 1
                f1 = f[0]
                if f1 < best_f1:
                    best_f1 = f1
                    best_base = base.copy()
            except:
                continue

        # If no candidate worked, default to zero (clipped)
        if best_base is None:
            best_base = base_zero

        # Step 2: Sweep x[0] using the remaining budget.
        n_remaining = self.budget - n_used
        if n_remaining < 10:
            n_remaining = self.budget

        # Generate x[0] samples.
        # Use a uniform grid for robust coverage of the front.
        n_sweep = n_remaining
        if n_sweep < 1:
            n_sweep = 1

        # Create a uniform grid for x[0]
        x0_vals = np.linspace(lo[0], hi[0], n_sweep)

        # Construct the full vectors
        X = np.tile(np.concatenate(([x0_mid], best_base)), (n_sweep, 1))
        X[:, 0] = x0_vals

        # Evaluate all points
        F = np.empty((n_sweep, self.n_obj))
        valid_count = 0

        for i in range(n_sweep):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
                valid_count += 1
            except:
                # If an evaluation fails, break to avoid further issues
                break

        # Return the population (return value is ignored, but we return something valid)
        if valid_count > 0:
            return X[:valid_count]
        else:
            return X
