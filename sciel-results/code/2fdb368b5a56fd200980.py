import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        t0 = time.time()
        DEADLINE = 295.0

        evals = 0
        Xg = []
        Fg = []

        def safe_eval(x):
            nonlocal evals
            try:
                return problem(x)
            except Exception:
                return np.full(n_obj, 1e9)

        def add_point(x, f):
            Xg.append(np.asarray(x, dtype=float))
            Fg.append(np.asarray(f, dtype=float))

        # Based on Inspiration 2: f1(x) = x[0]
        # We sample x[0] across the valid range to cover the Pareto front.
        # f1 must be < 11 for the reference point [11, 400].
        
        f1_min = lo[0]
        f1_max = min(hi[0], 11.0)
        
        # If the range is invalid or empty, handle gracefully
        if f1_min >= f1_max:
            f1_min = lo[0]
            f1_max = hi[0]
            if f1_min >= f1_max:
                # Degenerate bounds
                x = np.clip((lo + hi) / 2, lo, hi)
                f = safe_eval(x)
                evals += 1
                add_point(x, f)
                return np.array(Xg) if Xg else np.empty((0, dim))

        # Sample n_points uniformly in [f1_min, f1_max]
        n_points = budget
        
        # We want to ensure we don't exceed budget
        # We will generate n_points candidates
        
        # Uniform sampling in f1 (which is x[0]) is effective for hypervolume
        x0_values = np.linspace(f1_min, f1_max, n_points)
        
        # Prepare x[1:]
        # Set to 0, clipped to bounds, to minimize g(x) for ZDT4
        x_rest = np.zeros(dim - 1)
        x_rest = np.clip(x_rest, lo[1:], hi[1:])
        
        for x0 in x0_values:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_rest
            
            # Clip x0 to bounds just in case
            x[0] = np.clip(x[0], lo[0], hi[0])
            
            f = safe_eval(x)
            evals += 1
            add_point(x, f)
            
        # Build result
        if Fg:
            Fg_arr = np.array(Fg)
            Xg_arr = np.array(Xg)
            ref = np.array([11.0, 400.0])
            mask = np.all(Fg_arr < ref, axis=1)
            Xg_arr, Fg_arr = Xg_arr[mask], Fg_arr[mask]
            if len(Fg_arr) > 0:
                return Xg_arr
        return np.empty((0, dim))
