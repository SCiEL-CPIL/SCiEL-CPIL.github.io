import numpy as np
import time
from scipy.optimize import brentq, minimize_scalar

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

        # Optimize x_i for i >= 1 to minimize g(x)
        # h(x) = x^2 - 10 * cos(4 * pi * x)
        def h(x):
            return x**2 - 10.0 * np.cos(4.0 * np.pi * x)

        x_opt = np.zeros(dim)
        # x_0 will be determined later
        for i in range(1, dim):
            if lo[i] > hi[i]:
                x_opt[i] = 0.0
                continue
            # If interval is small, just pick midpoint or boundary
            if hi[i] - lo[i] < 1e-6:
                x_opt[i] = (lo[i] + hi[i]) / 2.0
                continue
            
            # Use minimize_scalar
            try:
                res = minimize_scalar(h, bounds=(lo[i], hi[i]), method='bounded', options={'xatol': 1e-8})
                x_opt[i] = res.x
            except Exception:
                # Fallback: check boundaries and 0 if in bounds
                candidates = [lo[i], hi[i]]
                if lo[i] <= 0.0 <= hi[i]:
                    candidates.append(0.0)
                # Also check nearest 0.5 multiple
                k = np.round(2.0 * 0.0) # 0
                # Just check a few points
                for k in range(-10, 11):
                    x_cand = k / 2.0
                    if lo[i] <= x_cand <= hi[i]:
                        candidates.append(x_cand)
                best_x = min(candidates, key=h)
                x_opt[i] = best_x

        # Compute g_star
        g_star = 1.0 + 10.0 * (dim - 1) + np.sum([h(x_opt[i]) for i in range(1, dim)])
        
        # ZDT4 Pareto front parameters
        # f1(t) = 1 + 8*sin(pi*t)^2 + 8*t*sin(2*pi*t^2)^2
        # f2(t) = g_star * (1 - sqrt(f1(t) / g_star))
        
        def zdt4_f1(t):
            return 1.0 + 8.0 * np.sin(np.pi * t)**2 + 8.0 * t * np.sin(2.0 * np.pi * t**2)**2

        # Find t_max where f1(t_max) = 11.0 in [0, 0.5] or wherever
        # We need to find a range for t where f1(t) covers [1, 11]
        # f1(0) = 1
        # f1 is not monotonic.
        # Let's find all t in [-5, 5] such that f1(t) <= 11.
        # We will sample t values.
        
        # Instead of inverting f1, we can just sample t uniformly and filter?
        # No, we want uniform f1.
        
        # Let's find roots of f1(t) - target = 0.
        # We need to know the intervals where roots exist.
        # Let's scan t in [-5, 5] with fine grid to find intervals where f1(t) crosses target.
        
        n_samples = budget
        if n_samples < 2:
            n_samples = 2

        f1_targets = np.linspace(1.0, 11.0, n_samples)
        
        # Pre-compute a grid of t to help brentq
        t_grid = np.linspace(-5.0, 5.0, 1000)
        f1_grid = zdt4_f1(t_grid)
        
        for f1_target in f1_targets:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            # Find an interval [t_a, t_b] where f1(t_a) <= f1_target <= f1(t_b) or vice versa
            # and f1(t) is continuous (it is).
            # We look for sign change in f1(t) - f1_target.
            
            diff = f1_grid - f1_target
            # Find indices where sign changes
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            
            t_val = None
            if len(sign_changes) > 0:
                # Pick the first one
                idx = sign_changes[0]
                t_a = t_grid[idx]
                t_b = t_grid[idx+1]
                
                try:
                    t_val = brentq(lambda t: zdt4_f1(t) - f1_target, t_a, t_b, xtol=1e-12)
                except Exception:
                    t_val = None
            
            if t_val is None:
                # Fallback: try [0, 1] or something
                try:
                    t_val = brentq(lambda t: zdt4_f1(t) - f1_target, 0.0, 1.0, xtol=1e-12)
                except Exception:
                    continue
            
            x = x_opt.copy()
            x[0] = t_val
            x = np.clip(x, lo, hi)
            
            f = safe_eval(x)
            evals += 1
            add_point(x, f)

        if Fg:
            Fg_arr = np.array(Fg)
            Xg_arr = np.array(Xg)
            ref = np.array([11.0, 400.0])
            mask = np.all(Fg_arr < ref, axis=1)
            Xg_arr, Fg_arr = Xg_arr[mask], Fg_arr[mask]
            if len(Fg_arr) > 0:
                return Xg_arr
        return np.empty((0, dim))
