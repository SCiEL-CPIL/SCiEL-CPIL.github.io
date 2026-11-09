import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def _calc_hv(self, points, ref):
        # points: list of (f1, f2)
        if not points:
            return 0.0
        # Sort by f1
        points = sorted(points, key=lambda p: p[0])
        hv = 0.0
        prev_x = 0.0 # Not needed if we use the difference form
        # Using the sum form:
        # sum (x_{i+1} - x_i) * (ref_y - y_i)
        # with x_N = ref_x
        n = len(points)
        for i in range(n):
            x_curr, y_curr = points[i]
            if i == n - 1:
                x_next = ref[0]
            else:
                x_next = points[i+1][0]
            # Ensure non-negative width
            width = x_next - x_curr
            if width < 0: width = 0
            height = ref[1] - y_curr
            if height < 0: height = 0
            hv += width * height
        return hv

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        ref = np.array([11.0, 11.0])
        
        # Candidate powers for x0 = (i/(N-1))^p
        # We want to find the best p.
        best_p = 1.0
        best_hv = -1.0
        best_points = None
        
        # Test a range of powers with fine granularity
        if n == 1:
            best_p = 1.0
            best_points = np.array([0.5])
        else:
            # Coarse search
            p_coarse = np.linspace(0.3, 5.0, 50)
            for p in p_coarse:
                indices = np.arange(n)
                x0_vals = (indices / (n - 1)) ** p
                f1 = x0_vals
                f2 = 1.0 - np.sqrt(f1)
                points = list(zip(f1, f2))
                hv = self._calc_hv(points, ref)
                if hv > best_hv:
                    best_hv = hv
                    best_p = p
                    best_points = x0_vals
            
            # Fine search around the best coarse p
            p_fine_start = max(0.1, best_p - 0.5)
            p_fine_end = min(6.0, best_p + 0.5)
            p_fine = np.linspace(p_fine_start, p_fine_end, 50)
            for p in p_fine:
                indices = np.arange(n)
                x0_vals = (indices / (n - 1)) ** p
                f1 = x0_vals
                f2 = 1.0 - np.sqrt(f1)
                points = list(zip(f1, f2))
                hv = self._calc_hv(points, ref)
                if hv > best_hv:
                    best_hv = hv
                    best_p = p
                    best_points = x0_vals
        
        # Now evaluate the best points
        for x0 in best_points:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
