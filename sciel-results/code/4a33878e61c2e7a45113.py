import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Strategy:
        # 1. Initial coarse search using L-BFGS-B from origin to find a good basin.
        # 2. Coordinate Descent with Golden Section Search to exploit anisotropy of Bent Cigar.
        #    This decouples dimensions, allowing precise convergence along the narrow valley.
        # 3. Final Nelder-Mead polish if budget remains.

        # --- Phase 1: L-BFGS-B ---
        # Use a fraction of the budget to get close to the minimum.
        # L-BFGS-B is efficient but can struggle with extreme ill-conditioning if not started well.
        # Origin is a safe start for BBOB f12.
        
        def lbfgsb_func(x):
            if self.calls >= self.budget or time.time() > deadline - 5:
                return 1e10
            f_val = self._evaluate(x)
            if f_val is None:
                return 1e10
            return f_val

        x0 = np.zeros(d)
        
        # Evaluate initial point
        f0 = self._evaluate(x0)
        if f0 is None:
             return np.zeros(d), 0.0

        # Run L-BFGS-B with a limited budget to avoid consuming everything
        # maxiter is a soft limit; we rely on call count via wrapper
        # We limit the number of iterations to ensure we have budget for CD
        try:
            res = minimize(lbfgsb_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * d, 
                           options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
        except Exception:
            pass

        # --- Phase 2: Coordinate Descent with Golden Section Search ---
        # This is the structural change. Bent Cigar is y1^2 + sum(1000*(yi)^2).
        # The valley is extremely narrow in dimensions 2..d.
        # Standard methods like NM or L-BFGS-B might take many steps to converge along the narrow axis.
        # Coordinate descent optimizes one axis at a time.
        # By using Golden Section Search for each axis, we can converge to high precision
        # with a predictable number of function evaluations per axis.
        
        if self.best_x is None:
            self.best_x = x0.copy()
            self.best_f = f0

        # Determine how many iterations of CD we can afford.
        # Each full CD pass costs d * (evals_per_golden).
        # Golden section search typically takes ~20-30 evals to converge to 1e-8 precision.
        # Let's aim for a few passes.
        
        # We will use the remaining budget for CD.
        # We need to be careful with the time limit.
        
        # Adaptive step size for Golden Section Search bounds.
        # We start with a wide range and shrink it if we hit the boundary.
        
        current_x = self.best_x.copy()
        current_f = self.best_f
        
        # Number of CD passes. We can run as many as budget allows.
        # Each pass: for each dim, do a 1D minimization.
        # 1D minimization via Golden Section:
        #   We need to find the min of f(x) along axis i.
        #   We define a local search window around current_x[i].
        
        # To be robust, we'll use a fixed number of passes based on remaining budget.
        # Let's say we reserve 10% of remaining budget for potential NM polish.
        
        # Calculate approximate cost per CD pass.
        # Golden section search convergence:
        # Interval length reduces by ~0.618 each step.
        # To reduce from size S to size eps: N ~ log(eps/S) / log(0.618).
        # If S=10, eps=1e-8: N ~ log(1e-9) / log(0.618) ~ -20.7 / -0.48 ~ 43 steps.
        # Each step is 1 eval (after initial 2). So ~45 evals per dim.
        # For d=5, one pass is ~225 evals.
        # With 500 budget, L-BFGS-B might use ~100-200.
        # Remaining ~300. So we can do 1-2 full passes.
        
        # Let's implement a helper for 1D Golden Section Minimization
        
        def golden_section_1d(axis, x_center, f_center, search_radius):
            """
            Minimizes func along 'axis' centered at x_center.
            Returns best_x_1d, best_f_1d.
            Uses self._evaluate which updates global best.
            """
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2
            inv_gr = 1.0 / gr
            
            # Initial interval
            a = x_center[axis] - search_radius
            b = x_center[axis] + search_radius
            
            # Clip to bounds
            a = max(-5.0, a)
            b = min(5.0, b)
            
            if b - a < 1e-12:
                return x_center[axis], f_center

            # Initial points
            c = b - inv_gr * (b - a)
            d_point = a + inv_gr * (b - a)
            
            x_c = x_center.copy()
            x_c[axis] = c
            f_c = self._evaluate(x_c)
            
            x_d = x_center.copy()
            x_d[axis] = d_point
            f_d = self._evaluate(x_d)
            
            if f_c is None or f_d is None:
                return x_center[axis], f_center
            
            # Keep track of best
            best_val = min(f_c, f_d, f_center)
            best_pos = x_center[axis]
            if f_c < best_val:
                best_val = f_c
                best_pos = c
            if f_d < best_val:
                best_val = f_d
                best_pos = d_point

            # Iterative refinement
            max_iter = 50
            for _ in range(max_iter):
                if self.calls >= self.budget or time.time() > deadline - 2:
                    break
                
                if f_c < f_d:
                    b = d_point
                    d_point = c
                    f_d = f_c
                    c = b - inv_gr * (b - a)
                    x_c = x_center.copy()
                    x_c[axis] = c
                    f_c = self._evaluate(x_c)
                    if f_c is None:
                        break
                    if f_c < best_val:
                        best_val = f_c
                        best_pos = c
                else:
                    a = c
                    c = d_point
                    f_c = f_d
                    d_point = a + inv_gr * (b - a)
                    x_d = x_center.copy()
                    x_d[axis] = d_point
                    f_d = self._evaluate(x_d)
                    if f_d is None:
                        break
                    if f_d < best_val:
                        best_val = f_d
                        best_pos = d_point
                
                # Convergence check
                if abs(b - a) < 1e-10:
                    break
            
            # Update global best if we found something better
            if best_val < self.best_f:
                self.best_f = best_val
                new_x = x_center.copy()
                new_x[axis] = best_pos
                self.best_x = new_x
            
            return best_pos, best_val

        # Run Coordinate Descent passes
        # We'll do multiple passes.
        # Pass 1: Wide search radius to ensure we are in the valley.
        # Pass 2+: Narrower radius for precision
        
        # We estimate the cost of a full CD pass.
        # For each dim, we do ~40-50 evals.
        # Total cost ~ 5 * 50 = 250.
        # We can afford 1 or 2 passes depending on remaining budget.
        
        # Let's do 2 passes.
        # Pass 1: radius 1.0
        # Pass 2: radius 0.1
        # If budget allows, Pass 3: radius 0.01
        
        radii = [1.0, 0.1, 0.01]
        
        for radius in radii:
            if self.calls >= self.budget - 100 or time.time() > deadline - 10:
                break
            
            for _ in range(2): # 2 passes per radius? Or just 1?
                # Let's do 1 pass per radius first to save budget for NM
                pass_flag = False
                for axis in range(d):
                    if self.calls >= self.budget - 50 or time.time() > deadline - 5:
                        break
                    
                    best_pos, best_val = golden_section_1d(axis, current_x, current_f, radius)
                    current_x[axis] = best_pos
                    current_f = best_val
                    # Update global best_x if this is better
                    if best_val < self.best_f:
                        self.best_f = best_val
                        self.best_x = current_x.copy()
            
            # Shrink radius for next pass
            # radius = radius * 0.1
            # But we have fixed radii list

        # --- Phase 3: Nelder-Mead Polish ---
        # If we have budget left, use NM to fine-tune.
        # NM is good for final convergence but can be slow.
        
        if self.calls < self.budget - 10 and time.time() < deadline - 10:
            
            def nm_func(x):
                if self.calls >= self.budget or time.time() > deadline - 2:
                    return 1e10
                f_val = self._evaluate(x)
                if f_val is None:
                    return 1e10
                return f_val

            x_start = self.best_x if self.best_x is not None else x0
            
            try:
                res2 = minimize(nm_func, x_start, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
            except Exception:
                pass

        # Return the best found
        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
            
        return self.best_x, self.best_f
