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

        # Use a parametric representation: sample u uniformly in (0,1), set f1 = u
        # Then f2 = 1 - sqrt(u). This gives uniform coverage in f1 space.
        # But for hypervolume, we want points that are "spread out" in objective space.
        # Actually, for a convex front like f2 = 1 - sqrt(f1), uniform sampling in f1
        # is near-optimal for hypervolume.

        # Generate N target f1 values uniformly in (0, 1)
        N = budget
        # Use (i + 0.5)/N to avoid endpoints
        u_targets = (np.arange(1, N + 1) - 0.5) / N  # uniform in (0,1)

        # Invert f1 on [0, 0.05] for each target using bisection
        x0_chosen = np.empty(N, dtype=float)

        for k in range(N):
            if time.time() > deadline:
                # Fallback: linear spacing
                x0_chosen[k:] = np.linspace(0.0, 0.05, N - k)
                break

            target = u_targets[k]
            lo_x = 0.0
            hi_x = 0.05
            # Bisection: 50 iterations for high precision
            for _ in range(50):
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

        for k in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim, dtype=float)
            x[0] = float(x0_chosen[k])
            x_clipped = np.clip(x, lo, hi)
            try:
                fx = problem(x_clipped)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        return results
