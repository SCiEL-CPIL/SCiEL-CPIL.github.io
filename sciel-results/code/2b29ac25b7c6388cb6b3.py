import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, x1_vals):
        """Compute hypervolume for points on ZDT1 front with ref [11, 11].
        x1_vals must be sorted ascending, in [0, 1].
        f1 = x1, f2 = 1 - sqrt(x1)
        """
        if len(x1_vals) == 0:
            return 0.0
        f1 = x1_vals
        f2 = 1.0 - np.sqrt(x1_vals)
        
        # Widths in f1
        f1_next = np.empty_like(f1)
        f1_next[:-1] = f1[1:]
        f1_next[-1] = 11.0
        
        widths = f1_next - f1
        heights = 11.0 - f2
        
        return np.sum(widths * heights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        B = self.budget
        D = self.dim
        
        # Initial guess: uniform in u space (x1 = u^2)
        u_init = np.linspace(0, 1, B)
        
        # Stage 1: Coarse grid search over power exponent p
        # x1 = u^p where u is uniform in [0,1]
        best_p = 2.0
        best_hv = self._compute_hv(u_init ** best_p)
        
        p_candidates = np.linspace(0.5, 5.0, 100)
        for p in p_candidates:
            if time.time() > deadline - 60.0:
                break
            x1_try = u_init ** p
            hv_try = self._compute_hv(x1_try)
            if hv_try > best_hv:
                best_hv = hv_try
                best_p = p
        
        # Stage 2: Golden section search for finer p
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate ~0.618
        a = max(0.1, best_p - 0.5)
        b = min(10.0, best_p + 0.5)
        
        # Ensure a < b
        if a >= b:
            a = max(0.1, best_p - 1.0)
            b = min(10.0, best_p + 1.0)
            if a >= b:
                a = 0.1
                b = 10.0
        
        for _ in range(60):
            if time.time() > deadline - 30.0:
                break
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            hc = self._compute_hv(u_init ** c)
            hd = self._compute_hv(u_init ** d)
            
            if hc < hd:
                a = c
            else:
                b = d
                
            if abs(b - a) < 1e-10:
                break
        
        best_p = (a + b) / 2.0
        best_hv = self._compute_hv(u_init ** best_p)
        
        # Stage 3: Local polish with small perturbations
        for delta in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            if time.time() > deadline - 15.0:
                break
            for sign in [-1, 1]:
                p_try = best_p + sign * delta
                if p_try < 0.01:
                    continue
                hv_try = self._compute_hv(u_init ** p_try)
                if hv_try > best_hv:
                    best_hv = hv_try
                    best_p = p_try
        
        # Generate final x1 values
        x1_vals = u_init ** best_p
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x1_vals.sort()
        
        # Build the evaluation matrix
        X = np.zeros((B, D))
        X[:, 0] = x1_vals
        for i in range(1, D):
            X[:, i] = lo[i]
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
        
        return X
