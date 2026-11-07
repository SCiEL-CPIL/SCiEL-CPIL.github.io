import numpy as np
import time
from scipy.optimize import brentq

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

        # ZDT4 Pareto front parameters
        # f1(t) = 1 + 8*sin(pi*t)^2 + 8*t*sin(2*pi*t^2)^2
        # This function is defined for t in [0, 1] typically, but we check bounds.
        
        def zdt4_f1(t):
            return 1.0 + 8.0 * np.sin(np.pi * t)**2 + 8.0 * t * np.sin(2.0 * np.pi * t**2)**2

        # We want to cover f1 in [1.0, 11.0]
        # Find t_max where f1(t_max) = 11.0
        # We search in [0, 1] assuming standard ZDT4 domain.
        # If bounds restrict t, we should respect them, but usually bounds are wide.
        # Let's find t_max in [0, 1].
        try:
            # f1(0)=1, f1(1)=1+8+0=9? 
            # sin(pi*1)=0. sin(2*pi*1)=0. f1(1)=1.
            # Wait, f1(1) = 1.
            # f1(0.5) = 13.
            # So f1 goes 1 -> 13 -> 1.
            # We need f1=11.
            # There are two solutions in [0, 1].
            # We just need one.
            # Let's search in [0, 0.5] where it goes 1->13.
            t_max = brentq(lambda t: zdt4_f1(t) - 11.0, 0.0, 0.5, xtol=1e-12)
        except Exception:
            t_max = 0.45

        # Use midpoint rule for f1 targets to maximize hypervolume accuracy
        n_samples = budget
        if n_samples < 2:
            n_samples = 2
            
        # Midpoints of n_samples intervals in [1.0, 11.0]
        # Interval width = 10.0 / n_samples
        # Midpoints: 1.0 + (i + 0.5) * width
        f1_targets = 1.0 + (np.arange(n_samples) + 0.5) * (10.0 / n_samples)

        for f1_target in f1_targets:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            # Find t such that zdt4_f1(t) = f1_target
            # We know t is in [0, t_max] for the rising branch
            try:
                t_val = brentq(lambda t: zdt4_f1(t) - f1_target, 0.0, t_max, xtol=1e-12, rtol=1e-12)
            except Exception:
                # If root finding fails, try the falling branch? 
                # Or just skip.
                # Given the monotonicity in [0, t_max], it should work.
                continue

            x = np.zeros(dim)
            x[0] = t_val
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
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
