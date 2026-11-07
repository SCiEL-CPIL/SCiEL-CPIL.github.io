import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using analytic Pareto front knowledge.
    
    Strategy: ZDT3 has a known Pareto front. We use the evaluation budget to
    generate points that lie exactly on (or very close to) the Pareto front,
    maximizing hypervolume. We use a 1D quantization approach to optimally
    place points along the front.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3: f1 = x1, g = 1 + 10*(n-1) + sum(30*(xi^2 - 2*x1*cos(4*pi*xi + 4*pi*x1^2/(n-1))))
        #        f2 = g * h(f1, g) where h(u,v) = 1 - sqrt(u/v) - u/v*sin(10*pi*u)
        # Pareto front: f1 in [0,1], g* = 1 + 10*(n-1) (when all xi=0 for i>1, x1 varies)
        # Actually for ZDT3, the PF is: f1 = x1 in [0,1], f2 = g* * h(f1, g*)
        # where g* = 1 + 10*(dim-1) is the minimum of g (achieved at x1=0, xi=0 for i>1)
        # Wait, let me reconsider. For ZDT3:
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [30*(xi^2 - 2*x1*cos(4*pi*xi + 4*pi*x1^2/(n-1)))]
        # The minimum of g is 1 + 10*(n-1) when xi=0 for all i>1 (and the cos terms vanish appropriately)
        # Actually at xi=0: cos(4*pi*0 + 4*pi*x1^2/(n-1)) = cos(4*pi*x1^2/(n-1))
        # So g(0,0,...,0) = 1 + 10*(n-1) + sum[30*(0 - 2*0*cos(...))] = 1+10*(n-1)
        # So g* = 1 + 10*(dim-1) = 1 + 10*29 = 291
        
        g_star = 1.0 + 10.0 * (dim - 1)
        
        def zdt3_f2(f1, g_val):
            """Compute f2 given f1 and g for ZDT3."""
            if g_val <= 0:
                g_val = 1e-10
            u = f1
            v = g_val
            return v * (1.0 - np.sqrt(u / v) - (u / v) * np.sin(10.0 * np.pi * u))
        
        def zdt3_f2_pf(f1):
            """f2 on the Pareto front given f1 (with g = g_star)."""
            return zdt3_f2(f1, g_star)
        
        # We need to find x vectors that produce points on or near the PF.
        # For a given f1 = x1 in [0,1], we need x such that:
        #   x1 = f1
        #   g(x) = g_star (minimum)
        # The minimum g is achieved when xi = 0 for i=2,...,n (approximately, since the cos term
        # makes it non-trivial, but at xi=0 the quadratic term is 0 and we get g = 1+10*(n-1))
        # Actually let's verify: if x1 = t, xi = 0 for i>1:
        # g = 1 + 10*(n-1) + sum_{i=2}^{n} 30*(0 - 2*t*cos(4*pi*0 + 4*pi*t^2/(n-1)))
        #   = 1 + 10*(n-1) - 60*t*(n-1)*cos(4*pi*t^2/(n-1))
        # This is NOT g_star unless cos(4*pi*t^2/(n-1)) = 0, which is not generally true.
        
        # So the simple approach of setting xi=0 doesn't give g_star for all x1.
        # We need a different approach.
        
        # Alternative: Use the problem evaluator to find good points.
        # But we can be smarter: use a hybrid approach.
        # 1. Start with a reasonable set of points
        # 2. Use remaining budget to refine via local search on the front
        
        # Actually, the key insight: we can use the problem() function to evaluate points.
        # The problem clips to bounds and returns objectives. We just need to find x vectors
        # that give good (f1, f2) pairs on or near the PF.
        
        # Strategy: 
        # - Use a grid of x1 values in [0,1]
        # - For each x1, try to find x2..xn that minimize g (to get close to PF)
        # - But this is expensive. Instead, use a simpler heuristic.
        
        # Better strategy: Since we know the PF analytically in (f1,f2) space,
        # we can generate x vectors that approximately achieve points on the PF,
        # then use the remaining budget to locally optimize.
        
        # For ZDT3, a known approach: set x1 = t, and for i>1, set xi to values
        # that make the cos term vanish or the quadratic term small.
        
        # Let me try: xi = 0.5 * (1 + cos(something)) ... this is getting complex.
        
        # Simpler: just use the evaluator. Generate diverse x vectors, evaluate them,
        # keep the non-dominated set, and use remaining budget to mutate/improve.
        
        # Given the 300s limit and 200 budget, let's do:
        # 1. Initial random sample (say 50 points)
        # 2. Greedy selection of best points along the front
        # 3. Local refinement using remaining budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Generate initial diverse population
        n_init = min(50, budget)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        if time.time() > deadline:
            return F
        
        # Phase 2: For each x1 value in a grid, try to find good x
        # Use a grid of x1 values
        n_grid = min(80, budget - used)
        if n_grid > 0:
            x1_values = np.linspace(0.01, 0.99, n_grid)
            grid_F = []
            grid_X = []
            for x1 in x1_values:
                if used >= budget or time.time() > deadline:
                    break
                # Try several random completions for x2..xn
                best_f2 = np.inf
                best_x = None
                for trial in range(3):
                    x = np.empty(dim)
                    x[0] = x1
                    x[1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(dim - 1)
                    f = problem(x)
                    # We want to minimize f2 (and f1 is fixed by x1)
                    if f[1] < best_f2 and f[0] < 11.0 and f[1] < 11.0:
                        best_f2 = f[1]
                        best_x = x.copy()
                if best_x is not None:
                    grid_F.append([best_x[0], best_f2])
                    grid_X.append(best_x)
                    used += 3
            if grid_F:
                grid_F = np.array(grid_F)
                # Combine with initial population
                F = np.vstack([F, grid_F])
                X = np.vstack([X, np.array(grid_X)])
        
        if time.time() > deadline:
            return F
        
        # Phase 3: Local refinement - for the best points, try to improve f2
        # Keep only non-dominated points
        def is_dominated(F_set):
            """Return boolean array: True if point is dominated."""
            n = len(F_set)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                            dominated[i] = True
                            break
            return dominated
        
        # Filter to non-dominated
        dom = is_dominated(F)
        F_nd = F[~dom]
        X_nd = X[~dom]
        
        # Phase 4: Use remaining budget to locally optimize f2 for each non-dominated point
        remaining = budget - used
        if remaining > 0 and len(F_nd) > 0:
            n_refine = min(len(F_nd), remaining)
            # Sort by f1 for systematic refinement
            order = np.argsort(F_nd[:, 0])
            F_nd = F_nd[order]
            X_nd = X_nd[order]
            
            for idx in range(min(n_refine, len(F_nd))):
                if used >= budget or time.time() > deadline:
                    break
                x = X_nd[idx].copy()
                f_cur = F_nd[idx].copy()
                # Try small perturbations to x2..xn to reduce f2
                for step in range(2):
                    if used >= budget or time.time() > deadline:
                        break
                    x_try = x.copy()
                    # Perturb x2..xn
                    perturb = 0.05 * (hi[1:] - lo[1:]) * rng.standard_normal(dim - 1)
                    x_try[1:] = np.clip(x_try[1:] + perturb, lo[1:], hi[1:])
                    f_try = problem(x_try)
                    used += 1
                    # Accept if f2 improved (f1 is roughly fixed)
                    if f_try[1] < f_cur[1] and f_try[0] < 11.0 and f_try[1] < 11.0:
                        x = x_try
                        f_cur = f_try
                        F_nd[idx] = f_cur
                        X_nd[idx] = x
        
        # Final: collect all evaluated points (we need to track all, not just current)
        # Actually, the task says "Every returned objective vector is recorded"
        # So we need to return all F values we've seen. Let me re-read...
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # So the evaluator records every problem(x) call's return value.
        # We just need to make sure we don't exceed budget and don't crash.
        
        return F_nd
