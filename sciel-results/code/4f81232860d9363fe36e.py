import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # Safety margin
        n = self.dim
        B = self.budget
        
        calls = 0
        best_x = np.zeros(n)
        best_f = float('inf')
        points = []
        values = []
        
        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B or (time.time() - start) > deadline:
                return float('inf')
            calls += 1
            y = np.clip(x, -5.0, 5.0)
            f = func(y)
            points.append(y.copy())
            values.append(f)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return f

        # Initial Sampling: Latin Hypercube
        rng = np.random.default_rng(42)
        N_init = min(20, B)
        for i in range(N_init):
            # Latin Hypercube Sampling
            x = np.zeros(n)
            for j in range(n):
                x[j] = -5.0 + 10.0 * (i + rng.random()) / N_init
            ev(x)
            
        # RBF Surrogate Parameters
        # We will use a subset of points as centers to keep the linear system small
        MAX_CENTERS = 50
        SIGMA = 1.0  # RBF width
        
        def build_rbf_model():
            if len(points) < 2:
                return None, None
            # Select centers: best points
            idx = np.argsort(values)
            idx = idx[:MAX_CENTERS]
            centers = np.array(points[i] for i in idx)
            target_vals = np.array(values[i] for i in idx)
            
            # Build kernel matrix
            M = len(centers)
            K = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    diff = centers[i] - centers[j]
                    K[i, j] = np.exp(-np.sum(diff**2) / (2 * SIGMA**2))
            
            # Regularize
            K += 1e-6 * np.eye(M)
            
            try:
                weights = np.linalg.solve(K, target_vals)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(K, target_vals, rcond=None)[0]
                
            return centers, weights

        def rbf_predict(x, centers, weights):
            diff = x - centers
            dists = np.sum(diff**2, axis=1)
            kernel_vals = np.exp(-dists / (2 * SIGMA**2))
            return np.sum(weights * kernel_vals)

        def rbf_variance(x, centers, weights):
            # Simple heuristic: variance is proportional to distance from centers
            # Or use the residual of the fit
            # For now, use a simple distance-based uncertainty
            dists = np.sum((x - centers)**2, axis=1)
            min_dist = np.min(dists)
            # Uncertainty increases with distance from nearest center
            return 1.0 + min_dist / (2 * SIGMA**2)

        def expected_improvement(x, centers, weights):
            mu = rbf_predict(x, centers, weights)
            sigma = rbf_variance(x, centers, weights) * 0.1  # Scale variance
            if sigma < 1e-8:
                return 0.0
            z = (best_f - mu) / sigma
            # EI = (best_f - mu) * Phi(z) + sigma * phi(z)
            # Approximate Phi and phi
            from scipy.stats import norm
            return (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        def acquisition_neg(x):
            if centers is None:
                return 0.0
            return -expected_improvement(x, centers, weights)

        # Main Loop
        while calls < B and (time.time() - start) < deadline:
            # Build model
            centers, weights = build_rbf_model()
            if centers is None:
                break
                
            # Find next point by maximizing EI
            best_acq_x = best_x.copy()
            best_acq_val = acquisition_neg(best_x)
            
            # Multi-start local search
            starts = [best_x]
            # Add some random starts
            for _ in range(5):
                starts.append(rng.uniform(-5, 5, size=n))
            # Add points from the current best neighborhood
            for _ in range(5):
                starts.append(best_x + rng.normal(0, 1.0, size=n))
                
            for start_pt in starts:
                start_pt = np.clip(start_pt, -5, 5)
                res = minimize(acquisition_neg, start_pt, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_acq_x = res.x
                    
            next_x = np.clip(best_acq_x, -5, 5)
            ev(next_x)
            
        return best_x, best_f
