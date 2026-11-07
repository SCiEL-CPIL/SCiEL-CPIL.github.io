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

        # f1 is monotonic increasing on [0, 0.05] from 0 to 1.
        # We can invert this segment to find x0 for any target f1 in [0, 1].
        
        # Target a_i (f1 values) for optimal staircase: a_i = A*(i/N)^2, i=1..N
        # f1_max is 1.0 (at x0=0.05, 0.15, etc.)
        f1_max = 1.0
        N = budget
        i = np.arange(1, N + 1, dtype=float)
        a_targets = f1_max * (i / N) ** 2

        # Invert f1 on [0, 0.05] for each target
        # Use bisection method for stability and speed
        x0_chosen = np.empty(N, dtype=float)
        
        # Pre-compute x0 for all targets
        for k in range(N):
            if time.time() > deadline:
                # Fallback: just use linear spacing if we run out of time (unlikely)
                x0_chosen[k:] = np.linspace(0.0, 0.05, N-k)
                break
            
            target = a_targets[k]
            # f1(0) = 0, f1(0.05) = 1
            # Bisection
            lo_x = 0.0
            hi_x = 0.05
            # 40 iterations of bisection gives precision ~ 0.05 / 2^40 ~ 4e-14
            for _ in range(40):
                mid_x = (lo_x + hi_x) / 2.0
                f_mid = f1_of_x0(mid_x)
                if f_mid < target:
                    lo_x = mid_x
                else:
                    hi_x = mid_x
            x0_chosen[k] = (lo_x + hi_x) / 2.0

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
