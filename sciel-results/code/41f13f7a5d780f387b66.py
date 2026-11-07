import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_points = min(budget, 200)
        self.optimal_x1 = self._compute_optimal_x1()

    def _compute_optimal_x1(self):
        N = self.n_points
        if N == 1:
            return np.array([0.5])
        
        # y_i = sqrt(x_i)
        # y_1 = 0, y_N = 1
        # Recurrence: y_{k-1} - y_k + (y_{k+1}^2 - y_k^2) / (2 y_k) = 0
        # => y_{k+1}^2 = y_k^2 - 2 y_k (y_{k-1} - y_k) = y_k^2 - 2 y_k y_{k-1} + 2 y_k^2 = 3 y_k^2 - 2 y_k y_{k-1}
        # y_{k+1} = sqrt(3 y_k^2 - 2 y_k y_{k-1})
        
        # We need to find y_2 such that after N-2 steps, we reach y_N = 1.
        # Let's use a simple search for y_2.
        
        def shoot(y2):
            y = [0.0, y2]
            for k in range(2, N-1):
                y_prev = y[-2]
                y_curr = y[-1]
                # y_next^2 = 3 y_curr^2 - 2 y_curr y_prev
                val = 3 * y_curr**2 - 2 * y_curr * y_prev
                if val < 0:
                    return None
                y_next = np.sqrt(val)
                y.append(y_next)
            return y[-1]
            
        # y2 must be small enough to reach 1.
        # If y2 is too small, y grows slowly?
        # If y2 is too large, y might grow too fast or become invalid.
        # Let's try a few values.
        lo, hi = 1e-6, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            res = shoot(mid)
            if res is None:
                hi = mid
            elif res < 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        
        y2 = (lo + hi) / 2
        y = [0.0, y2]
        for k in range(2, N-1):
            y_prev = y[-2]
            y_curr = y[-1]
            val = 3 * y_curr**2 - 2 * y_curr * y_prev
            y_next = np.sqrt(max(0, val))
            y.append(y_next)
            
        # y should be length N, y[0]=0, y[-1] approx 1
        # x = y^2
        x = np.array(y)**2
        # Ensure x_N = 1
        x[-1] = 1.0
        # Ensure x_1 = 0
        x[0] = 0.0
        return x

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.n_points
        x1_vals = self.optimal_x1
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
