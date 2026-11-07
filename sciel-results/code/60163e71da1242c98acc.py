import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        return self.func(x)

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, leave 10s margin for safety
        deadline = start_time + 290.0 
        
        d = self.dim
        self.func = func
        self.calls = 0
        
        best_x = None
        best_f = float('inf')
        
        # Golden Section Search for 1D minimization
        # Returns (x_min, f_min)
        def golden_section_search(x_fixed, idx, lower, upper, tol=1e-8, max_iter=50):
            phi = (1 + 5**0.5) / 2
            invphi = 1 / phi
            invphi2 = 1 / (phi * phi)
            
            a = lower
            b = upper
            c = b - invphi * (b - a)
            d_pt = a + invphi * (b - a)
            
            x_c = x_fixed.copy()
            x_d = x_fixed.copy()
            x_c[idx] = c
            x_d[idx] = d_pt
            
            f_c = self._evaluate(x_c)
            f_d = self._evaluate(x_d)
            
            if f_c is None or f_d is None:
                return None, None

            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if self.calls >= self.budget:
                    break
                if b - a < tol:
                    break
                    
                if f_c < f_d:
                    b = d_pt
                    d_pt = c
                    f_d = f_c
                    c = b - invphi * (b - a)
                    x_c = x_fixed.copy()
                    x_c[idx] = c
                    f_c = self._evaluate(x_c)
                    if f_c is None:
                        break
                else:
                    a = c
                    c = d_pt
                    f_c = f_d
                    d_pt = a + invphi * (b - a)
                    x_d = x_fixed.copy()
                    x_d[idx] = d_pt
                    f_d = self._evaluate(x_d)
                    if f_d is None:
                        break
            
            # Final check
            if b - a < tol:
                x_mid = (a + b) / 2
                x_mid_vec = x_fixed.copy()
                x_mid_vec[idx] = x_mid
                f_mid = self._evaluate(x_mid_vec)
                if f_mid is not None:
                    if f_mid < f_c:
                        return x_mid_vec, f_mid
                    else:
                        return x_c, f_c
            
            # Return the better of the last two points
            if f_c < f_d:
                return x_c, f_c
            else:
                return x_d, f_d

        # Strategy: Multiple restarts with coordinate descent
        # Bent Cigar is unimodal, so a single good start is usually enough, 
        # but restarts guard against bad initializations or numerical issues.
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            
            # Initialize
            if r == 0:
                # Start at origin, which is often close to the minimum in BBOB
                x = np.zeros(d)
            elif r == 1:
                # Start at a random point
                rng = np.random.default_rng(42 + r)
                x = rng.uniform(-4, 4, d)
            else:
                # Start near previous best if available, or random
                if best_x is not None:
                    x = best_x.copy()
                else:
                    rng = np.random.default_rng(100 + r)
                    x = rng.uniform(-4, 4, d)
            
            f = self._evaluate(x)
            if f is None:
                continue
                
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Coordinate Descent Loop
            # We iterate over coordinates. If we improve, we continue.
            # We stop if we make no progress in a full cycle.
            max_cycles = 20
            for cycle in range(max_cycles):
                if self.calls >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                improved = False
                for i in range(d):
                    if self.calls >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    # Perform 1D search along dimension i
                    # Bounds are [-5, 5]
                    # We can tighten bounds if we want, but [-5,5] is safe.
                    # To save calls, we can limit the number of iterations in GSS.
                    # Budget check: GSS takes ~2 calls initially + 2 per iter.
                    # With 500 calls total, we can afford ~250 GSS iterations total.
                    # 5 dims * 20 cycles * ~10 GSS iters/cycle/dim = 1000 calls (too many).
                    # So we need to be careful with max_iter in GSS.
                    
                    # Adaptive max_iter for GSS:
                    # If we have plenty of budget left, use more precision.
                    calls_left = self.budget - self.calls
                    # Estimate calls needed for this coordinate
                    # Let's aim to use at most 10% of remaining budget per coordinate per cycle?
                    # No, simpler: fixed reasonable max_iter.
                    # 20 cycles * 5 dims = 100 coordinates.
                    # 500 calls / 100 coords = 5 calls per coordinate.
                    # GSS: 2 init + 2*iters. 5 calls -> 1 iter. Too few.
                    # We need to reduce cycles or increase budget usage per coord.
                    
                    # Let's reduce cycles to 5 and use more precision per coordinate.
                    # 5 cycles * 5 dims = 25 coordinates.
                    # 500 / 25 = 20 calls per coordinate.
                    # GSS: 2 + 2*iters = 20 -> 9 iters. Good.
                    
                    x_new, f_new = golden_section_search(x, i, -5.0, 5.0, tol=1e-6, max_iter=10)
                    
                    if x_new is not None and f_new is not None:
                        if f_new < f - 1e-10:
                            x = x_new
                            f = f_new
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                
                if not improved:
                    break
        
        # Final Polish: If we have budget left, do a high-precision coordinate descent
        if best_x is not None and self.calls < self.budget and time.time() < deadline - 5:
            x = best_x.copy()
            f = best_f
            # High precision polish
            for i in range(d):
                if self.calls >= self.budget:
                    break
                if time.time() > deadline - 2:
                    break
                x_new, f_new = golden_section_search(x, i, -5.0, 5.0, tol=1e-9, max_iter=20)
                if x_new is not None and f_new is not None and f_new < f:
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

        if best_x is None:
            # Fallback
            best_x = np.zeros(d)
            best_f = self._evaluate(best_x)
            if best_f is None:
                best_f = 0.0

        return best_x, best_f
