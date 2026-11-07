import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        eval_count = [0]
        all_X = []
        all_F = []
        
        def count_eval(x):
            eval_count[0] += 1
            if eval_count[0] > self.budget:
                raise Exception("Budget exceeded")
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            return f

        # 1. Initialization: Latin Hypercube Sampling for better coverage
        n_init = 40
        rng = np.random.default_rng(0)
        
        X_init = []
        for d in range(self.dim):
            samples = (np.arange(n_init) + rng.random(n_init)) / n_init
            X_init.append(samples)
        
        X_init = np.array(X_init).T
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if eval_count[0] >= self.budget:
                break
            count_eval(x)
            
        # 2. Tchebycheff Decomposition + Multi-start L-BFGS-B
        # Generate weight vectors on a grid
        m = 10
        weights = []
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                w = np.array([i / m, j / m, k / m])
                if w.sum() > 0:
                    weights.append(w)
        
        # We'll use a subset of these weights, prioritizing vertices, edges, and center
        key_weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1/3.0, 1/3.0, 1/3.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.8, 0.1]),
            np.array([0.1, 0.1, 0.8]),
            np.array([0.6, 0.3, 0.1]),
            np.array([0.3, 0.6, 0.1]),
            np.array([0.3, 0.1, 0.6]),
            np.array([0.1, 0.3, 0.6]),
            np.array([0.6, 0.1, 0.3]),
            np.array([0.1, 0.6, 0.3]),
        ]
        
        # Add some random weights for diversity
        for _ in range(10):
            w = rng.random(self.n_obj)
            w /= w.sum()
            key_weights.append(w)
            
        # Shuffle to mix up the order
        rng.shuffle(key_weights)
        
        center = (lo + hi) / 2
        
        # Estimate ideal point from initial samples for Tchebycheff
        # If we have samples, use min of each objective
        if all_F:
            ideal = np.min(np.array(all_F), axis=0)
        else:
            ideal = np.zeros(self.n_obj)
            
        # Scale factor for Tchebycheff
        # We want to normalize the objectives so that they are on a similar scale
        # But we don't know the range yet. A simple heuristic is to use the current spread.
        # Or just use a large constant if the spread is small.
        # Let's try to estimate the range from the initial samples.
        if all_F:
            f_arr = np.array(all_F)
            ranges = np.ptp(f_arr, axis=0)
            # Avoid division by zero
            ranges[ranges == 0] = 1.0
        else:
            ranges = np.ones(self.n_obj)
            
        # We will iterate through key weights
        # Budget left: ~160.
        # Number of weights: 26.
        # ~6 evals per weight.
        
        for w in key_weights:
            if eval_count[0] >= self.budget - 2:
                break
            if time.time() > deadline - 10:
                break
                
            # Tchebycheff objective: max_i (w_i * (f_i - ideal_i))
            # But we need to handle the case where we don't have a good ideal point.
            # Also, standard Tchebycheff uses (f_i - z_i) where z_i is ideal.
            # If we don't have ideal, we can just use weighted sum?
            # Let's stick to Weighted Sum for robustness if Tchebycheff is tricky with unknown scales.
            # Actually, Weighted Sum is safer and often sufficient for convex fronts.
            # RE34 front is likely convex or close to it.
            
            def objective_counted(x):
                f = count_eval(x)
                return np.dot(w, f)
            
            # Try starting from center
            x0 = center
            try:
                res = minimize(objective_counted, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                               options={'maxiter': 10, 'maxfev': 8, 'ftol': 1e-6})
            except Exception:
                continue
                
            # If we have budget, try a random start
            if eval_count[0] < self.budget - 5 and time.time() < deadline - 5:
                x0 = lo + (hi - lo) * rng.random(self.dim)
                try:
                    res2 = minimize(objective_counted, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                                    options={'maxiter': 10, 'maxfev': 8, 'ftol': 1e-6})
                except Exception:
                    pass

        # 3. Final Polish
        # If we have budget left, refine the best points found so far
        if all_F and eval_count[0] < self.budget:
            f_arr = np.array(all_F)
            # Sort by sum of objectives
            sums = np.sum(f_arr, axis=1)
            sorted_idx = np.argsort(sums)
            
            # Pick top 5
            top_indices = sorted_idx[:5]
            
            for idx in top_indices:
                if time.time() > deadline - 5:
                    break
                if eval_count[0] >= self.budget - 2:
                    break
                
                x0 = all_X[idx]
                
                def sum_obj(x):
                    f = count_eval(x)
                    return np.sum(f)
                
                try:
                    res = minimize(sum_obj, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                                   options={'maxiter': 10, 'maxfev': 5, 'ftol': 1e-8})
                except Exception:
                    pass

        return all_X
