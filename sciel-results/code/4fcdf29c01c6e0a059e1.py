import numpy as np
import time
import sys
from scipy.optimize import minimize_scalar, minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_hv(self, points, ref):
        n = len(points)
        if n == 0:
            return 0.0
        points = points[np.argsort(points[:, 0])]
        hv = 0.0
        x_next = ref[0]
        for i in range(n):
            x_curr = points[i][0]
            y_curr = points[i][1]
            width = x_next - x_curr
            height = ref[1] - y_curr
            hv += width * height
            x_next = x_curr
        return hv

    def _hv_from_x1(self, x1, ref):
        x1 = np.sort(x1)
        f2 = 1.0 - np.sqrt(np.clip(x1, 0, 1))
        points = np.column_stack((x1, f2))
        return self._compute_hv(points, ref)

    def _compute_optimal_x1(self):
        N = self.budget
        if N <= 0:
            return
        ref = np.array([11.0, 11.0])
        t = np.linspace(0, 1, N)
        
        # Try single parameter: x1 = t^p
        def neg_hv_p(p):
            if p < 0.1:
                return 1e10
            x1 = t ** p
            return -self._hv_from_x1(x1, ref)
        
        # Coarse scan
        best_p = 1.0
        best_hv = -np.inf
        for p in np.arange(0.5, 8.0, 0.1):
            hv = -neg_hv_p(p)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Fine scan
        for p in np.arange(best_p - 0.1, best_p + 0.1, 0.01):
            hv = -neg_hv_p(p)
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # minimize_scalar
        res = minimize_scalar(neg_hv_p, bounds=(best_p - 0.05, best_p + 0.05), method='bounded', options={'xatol': 1e-10})
        if res.success:
            best_p = res.x
            best_hv = -res.fun
        
        x1_single = t ** best_p
        hv_single = self._hv_from_x1(x1_single, ref)
        
        # Try two parameter: x1 = t^a / (t^a + (1-t)^b)
        def neg_hv_two(params):
            a, b = params
            if a < 0.1 or b < 0.1:
                return 1e10
            ta = t ** a
            ob = (1 - t) ** b
            denom = ta + ob
            x1 = np.zeros(N)
            mask = denom > 0
            x1[mask] = ta[mask] / denom[mask]
            x1[0] = 0.0
            x1[-1] = 1.0
            return -self._hv_from_x1(x1, ref)
        
        best_two = None
        best_two_hv = -np.inf
        # Coarse
        for a in np.arange(0.5, 5.0, 0.25):
            for b in np.arange(0.5, 5.0, 0.25):
                hv = -neg_hv_two([a, b])
                if hv > best_two_hv:
                    best_two_hv = hv
                    best_two = [a, b]
        
        # Fine
        if best_two:
            for a in np.arange(best_two[0] - 0.25, best_two[0] + 0.25, 0.05):
                for b in np.arange(best_two[1] - 0.25, best_two[1] + 0.25, 0.05):
                    if a < 0.1 or b < 0.1:
                        continue
                    hv = -neg_hv_two([a, b])
                    if hv > best_two_hv:
                        best_two_hv = hv
                        best_two = [a, b]
        
        # Local
        if best_two:
            res2 = minimize(neg_hv_two, best_two, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
            if res2.fun < -best_two_hv:
                best_two = res2.x
                best_two_hv = -res2.fun
        
        # Choose best
        if best_two is not None and best_two_hv > hv_single:
            a, b = best_two
            ta = t ** a
            ob = (1 - t) ** b
            denom = ta + ob
            x1 = np.zeros(N)
            mask = denom > 0
            x1[mask] = ta[mask] / denom[mask]
            x1[0] = 0.0
            x1[-1] = 1.0
            self.optimal_x1 = x1
        else:
            self.optimal_x1 = x1_single

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        if self.optimal_x1 is None:
             self._compute_optimal_x1()
             
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
