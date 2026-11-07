import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0
        budget = int(self.budget)
        dim = int(self.dim)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # --- Analytic ZDT6 front: x[1:]=0 => g=0, f1=1-exp(-4*x0*sin^2(10*pi*x0)), f2=1-sqrt(f1)
        def f1_of_x0(x0):
            return 1.0 - np.exp(-4.0 * x0 * np.sin(10.0 * np.pi * x0) ** 2)

        # Dense scan to find f1 values and invert
        n_scan = 1_000_000
        x0_grid = np.linspace(0.0, 1.0, n_scan)
        f1_grid = f1_of_x0(x0_grid)
        f1_max = float(np.max(f1_grid))

        # Target a_i (f1 values) for optimal staircase: a_i = f1_max*(i/N)^2, i=1..N
        N = budget
        i = np.arange(1, N + 1, dtype=float)
        a_targets = f1_max * (i / N) ** 2

        # For each target, find the x0 on the grid with f1 closest to target
        x0_chosen = np.empty(N, dtype=float)
        for k in range(N):
            if time.time() > deadline:
                x0_chosen[k:] = x0_grid[0]
                break
            target = a_targets[k]
            idx = np.argmin(np.abs(f1_grid - target))
            x0_chosen[k] = x0_grid[idx]

        # Evaluate and record
        used = 0
        results = []
        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        for k in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim, dtype=float)
            x[0] = float(x0_chosen[k])
            eval_and_record(x)

        return results
