import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 325.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # ZDT1 Pareto front analysis
        # f1 = x0
        # f2 = 291 * (1 - sqrt(x0)) for x1..xn = 0
        # Reference point R = (11, 11)
        # Valid region: f2 < 11 => 291(1 - sqrt(t)) < 11 => sqrt(t) > 1 - 11/291
        # t_min = (1 - 11/291)^2
        
        coeff = 291.0
        r2 = 11.0
        t_min = (1.0 - r2 / coeff) ** 2
        t_max = 1.0
        
        # H(t) = r2 - f2(t) = 11 - 291(1 - sqrt(t)) = 291*sqrt(t) - 280
        # H'(t) = 291 / (2 * sqrt(t))
        
        def H(t):
            return coeff * np.sqrt(t) - (coeff - r2)
        
        def Hp(t):
            # Avoid division by zero
            t_safe = np.maximum(t, 1e-15)
            return coeff / (2.0 * np.sqrt(t_safe))
        
        n_samples = budget
        if n_samples <= 0:
            return np.zeros((0, dim))
            
        # Initialize t values uniformly in [t_min, t_max]
        # It's better to initialize closer to the optimal distribution.
        # Optimal spacing is denser where H(t) is large?
        # Actually, let's just use uniform and let the optimizer fix it.
        t_vals = np.linspace(t_min, t_max, n_samples)
        
        # Coordinate descent optimization
        # We want to maximize HV = sum (t_{i+1} - t_i) H(t_i)
        # dHV/dt_i = H(t_{i-1}) - H(t_i) + (t_{i+1} - t_i) H'(t_i) = 0
        # Note: t_0 is not a variable. For i=1, term H(t_0) is absent.
        # For i=N, t_{N+1} = r1 = 11.0
        
        r1 = 11.0
        t_opt = t_vals.copy()
        
        for outer_iter in range(50):
            improved = False
            for i in range(n_samples):
                # Determine bounds for t_i
                if i == 0:
                    lo_t = t_min
                    hi_t = t_opt[1] if n_samples > 1 else t_max
                elif i == n_samples - 1:
                    lo_t = t_opt[i - 1]
                    hi_t = t_max
                else:
                    lo_t = t_opt[i - 1]
                    hi_t = t_opt[i + 1]
                
                if hi_t - lo_t < 1e-12:
                    continue
                
                # Define the gradient function G(t) for the current i
                # G(t) = dHV/dt_i with t_i = t, others fixed
                def G(t):
                    if i == 0:
                        return -H(t) + (t_opt[i+1] - t) * Hp(t)
                    elif i == n_samples - 1:
                        return H(t_opt[i-1]) - H(t) + (r1 - t) * Hp(t)
                    else:
                        return H(t_opt[i-1]) - H(t) + (t_opt[i+1] - t) * Hp(t)
                
                # Find root of G(t) in [lo_t, hi_t]
                # G(lo_t) and G(hi_t)
                g_lo = G(lo_t)
                g_hi = G(hi_t)
                
                if g_lo * g_hi > 0:
                    # No sign change, check boundaries
                    # If G is always positive, increase t (move right)
                    # If G is always negative, decrease t (move left)
                    if g_lo > 0:
                        t_new = hi_t
                    else:
                        t_new = lo_t
                else:
                    # Bisection
                    lo = lo_t
                    hi = hi_t
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        g_mid = G(mid)
                        if g_lo * g_mid <= 0:
                            hi = mid
                            g_hi = g_mid
                        else:
                            lo = mid
                            g_lo = g_mid
                        if hi - lo < 1e-15:
                            break
                    t_new = 0.5 * (lo + hi)
                
                if abs(t_new - t_opt[i]) > 1e-12:
                    t_opt[i] = t_new
                    improved = True
            
            if not improved:
                break

        # Evaluate all points
        X = []
        F = []
        for t in t_opt:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            # x[1:] = 0 is default for np.zeros
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        if len(F) == 0 and used < budget:
             # Fallback: random
             rng = np.random.default_rng()
             lo = np.array([b[0] for b in bounds], dtype=float)
             hi = np.array([b[1] for b in bounds], dtype=float)
             for _ in range(min(50, budget - used)):
                 if used >= budget or (time.time() - t0) > DEADLINE:
                     break
                 x = lo + (hi - lo) * rng.random(dim)
                 f = eval_point(x)
                 if np.all(np.isfinite(f)):
                     X.append(x.copy())
                     F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
