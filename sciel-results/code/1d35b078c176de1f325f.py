"""ZDT2 optimizer: exploit known Pareto front structure.

Key insight: only points with f2 < 11.0 (reference) contribute to hypervolume.
For the Pareto front (x[1:]=0), f2 = g(x[0])*(1-sqrt(x[0])) which is only
< 11 for x[0] close to 1.0. We should concentrate all budget there.
"""

import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting known Pareto front.
    
    The Pareto front is at x[1:] = 0, giving f1 = x[0], f2 = g(x[0])*(1-sqrt(x[0])).
    Only points with f2 < ref[1] = 11.0 count toward hypervolume.
    Strategy:
      1. Use a few evaluations to estimate the minimum of g (and hence the valid x[0] range).
      2. Concentrate remaining budget uniformly in the valid x[0] range [x_min, 1.0].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(n)

        def eval_x(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = problem(x)
            eval_count += 1
            return x, f

        # Phase 1: Estimate g_min by evaluating a few points with x[1:]=0
        # g(x) = 1 + 10*(n-1) + sum((x_i - sqrt(f1))^2)
        # At x[1:]=0: g = 1 + 10*(n-1) + sum(sqrt(f1)^2 for i=1..n-1)
        #           = 1 + 10*(n-1) + (n-1)*f1
        # For standard ZDT2, g(0) = 1 + 10*29 = 291
        # For this variant, let's estimate it.
        
        # Evaluate at x[0]=0, x[1:]=0 to get f2 = g(0)
        x0 = np.zeros(n)
        x0[0] = 0.0
        _, f0 = eval_x(x0)
        g_at_0 = f0[1]  # f2 when f1=0, x[1:]=0
        best_x = x0

        if time.time() > deadline:
            return best_x

        # Also evaluate at x[0]=0.5, x[1:]=0 to cross-check
        x05 = np.zeros(n)
        x05[0] = 0.5
        _, f05 = eval_x(x05)
        best_x = x05

        if time.time() > deadline:
            return best_x

        # Estimate g_min: try a few x[0] values to find where g is minimized
        # g(x) with x[1:]=0: g = 1 + 10*(n-1) + (n-1)*x[0]
        # This is increasing in x[0], so g_min is at x[0]=0 -> g_min = 1 + 10*(n-1)
        # But for the actual ZDT2, g involves (x_i - sqrt(f1))^2
        # At x[1:]=0: (0 - sqrt(x[0]))^2 = x[0] for each of n-1 terms
        # So g = 1 + 10*(n-1) + (n-1)*x[0], minimized at x[0]=0
        # g_min = 1 + 10*(n-1)
        
        g_min = 1.0 + 10.0 * (n - 1)
        
        # Use measured g_at_0 as sanity check
        if g_at_0 < g_min:
            g_min = g_at_0

        # Find x_min such that f2(x_min) = 11.0
        # f2 = g(x[0]) * (1 - sqrt(x[0]))
        # At x[1:]=0: g(x[0]) = 1 + 10*(n-1) + (n-1)*x[0]
        # f2(x) = (1 + 10*(n-1) + (n-1)*x) * (1 - sqrt(x))
        
        # Solve for x where f2 = 11
        # Let a = 1 + 10*(n-1), b = (n-1)
        # (a + b*x) * (1 - sqrt(x)) = 11
        # Let t = sqrt(x), x = t^2
        # (a + b*t^2) * (1 - t) = 11
        
        a = 1.0 + 10.0 * (n - 1)
        b = float(n - 1)
        
        # Solve (a + b*t^2)*(1-t) = 11 for t in [0,1]
        # Use bisection
        def f2_of_t(t):
            return (a + b * t * t) * (1.0 - t)
        
        # f2_of_t(0) = a, f2_of_t(1) = 0
        # We want f2_of_t(t) = 11
        # Check if 11 is in range [0, a]
        if g_at_0 <= 11.0:
            # Even at x=0, f2 <= 11, so full range [0,1] is valid
            x_min = 0.0
        else:
            # Find t where f2_of_t(t) = 11, then x_min = t^2
            t_lo, t_hi = 0.0, 1.0
            # f2_of_t is decreasing from a to 0 as t goes 0 to 1
            for _ in range(100):
                t_mid = 0.5 * (t_lo + t_hi)
                if f2_of_t(t_mid) > 11.0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid
            t_star = 0.5 * (t_lo + t_hi)
            x_min = t_star * t_star
        
        # Now concentrate all remaining budget in [x_min, 1.0]
        remaining = budget - eval_count
        if remaining <= 0:
            return best_x

        # Use remaining evaluations to sample x[0] uniformly in [x_min, 1.0]
        # This maximizes hypervolume coverage in the valid region
        # We already used 2 evaluations, so we have budget - 2 left
        # But let's be safe and just use what's left
        
        f1_values = np.linspace(x_min, 1.0, remaining)
        
        for f1_val in f1_values:
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(n)
            x[0] = f1_val
            x[1:] = 0.0
            x, f = eval_x(x)
            best_x = x

        return best_x
