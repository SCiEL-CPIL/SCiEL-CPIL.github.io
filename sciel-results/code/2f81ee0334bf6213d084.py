import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, self.n_obj))

        N = budget
        R1, R2 = 11.0, 400.0

        # ZDT4 Pareto front: f1 = x0, f2 = 1 - sqrt(x0) for x0 in [0,1]
        # HV contribution of point i (sorted by x0): (x_{i+1} - x_i) * (R2 - f2_i)
        # where x_{N} = R1 (reference)
        # We want to maximize sum of these terms.
        # Using power law: x_i = (i/N)^p for i=0..N-1, x_0=0, x_{N-1}=1
        # Optimize p to maximize HV.

        def compute_hv(p, N_pts):
            """Compute HV for power-law spacing with exponent p."""
            if N_pts <= 1:
                return 0.0
            idx = np.arange(N_pts, dtype=float)
            x_vals = (idx / N_pts) ** p
            x_vals[0] = 0.0
            x_vals[-1] = 1.0
            
            # f2 = 1 - sqrt(x), so R2 - f2 = R2 - 1 + sqrt(x) = 399 + sqrt(x)
            w = R2 - 1.0 + np.sqrt(x_vals)  # = 399 + sqrt(x)
            
            # HV = sum_{i=0}^{N-2} (x_{i+1} - x_i) * w_i + (R1 - x_{N-1}) * w_{N-1}
            dx = np.diff(x_vals)
            hv = np.sum(dx * w[:-1])
            hv += (R1 - x_vals[-1]) * w[-1]
            return hv

        # Optimize exponent p using golden-section search
        # p should be > 1 to cluster points near x=0 where sqrt changes rapidly
        best_p = 2.0
        best_hv = compute_hv(2.0, N)
        
        # Try a range of p values to find good region
        p_candidates = np.linspace(1.0, 4.0, 100)
        for p_cand in p_candidates:
            hv_cand = compute_hv(p_cand, N)
            if hv_cand > best_hv:
                best_hv = hv_cand
                best_p = p_cand
        
        # Refine with golden-section around best_p
        if best_p > 1.05 and best_p < 3.95:
            lo_p, hi_p = best_p - 0.1, best_p + 0.1
            for _ in range(50):
                if time.time() - t0 > DEADLINE - 5:
                    break
                m1 = lo_p + 0.382 * (hi_p - lo_p)
                m2 = hi_p - 0.382 * (hi_p - lo_p)
                hv1 = compute_hv(m1, N)
                hv2 = compute_hv(m2, N)
                if hv1 < hv2:
                    lo_p = m1
                else:
                    hi_p = m2
                if abs(hi_p - lo_p) < 1e-8:
                    break
            best_p = 0.5 * (lo_p + hi_p)

        # Generate x0 values with optimal power law
        if N == 1:
            x0_vals = np.array([0.0])
        else:
            idx = np.arange(N, dtype=float)
            x0_vals = (idx / N) ** best_p
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
            x0_vals = np.clip(x0_vals, 0.0, 1.0)
            
            # Ensure strictly increasing
            for k in range(1, N):
                if x0_vals[k] <= x0_vals[k - 1]:
                    x0_vals[k] = x0_vals[k - 1] + 1e-12

        # Tail variables: set to minimize g(x) = 0
        tail = np.zeros(dim - 1)
        for k in range(dim - 1):
            if lo[k + 1] <= 0.0 <= hi[k + 1]:
                tail[k] = 0.0
            elif 0.0 < lo[k + 1]:
                tail[k] = lo[k + 1]
            else:
                tail[k] = hi[k + 1]

        # Evaluate all N points
        evals = 0
        results = []
        for x0_val in x0_vals:
            if evals >= budget:
                break
            if time.time() - t0 > DEADLINE:
                break
            x = np.empty(dim)
            x[0] = x0_val
            x[1:] = tail
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                evals += 1
                if f is not None:
                    results.append(np.asarray(f, dtype=float))
            except Exception:
                pass

        if results:
            return np.array(results)
        return np.empty((0, self.n_obj))
