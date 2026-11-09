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
        deadline = start_time + 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        X = []
        F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        # 1. Latin Hypercube Sampling for initial coverage
        # Use 60 points for global exploration (save budget for local search)
        n_lhs = 60
        if n_lhs > self.budget:
            n_lhs = self.budget
            
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            edges = np.linspace(0, 1, n_lhs + 1)
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points_in_bins)
            lhs_points[:, d] = points_in_bins
        
        # Scale to bounds
        lhs_points = lo + lhs_points * span
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = evaluate(x)
            X.append(x)
            F.append(fx)
            
        if n_evals >= self.budget or time.time() > deadline:
            return np.array(X)
            
        # 2. Local Search using scipy.optimize.minimize
        X = np.array(X)
        F = np.array(F)
        
        # Find non-dominated points
        def is_dominated(F, i):
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False
            
        nondom_indices = [i for i in range(len(F)) if not is_dominated(F, i)]
        
        if not nondom_indices:
            nondom_indices = list(range(len(F)))
            
        # Select top K non-dominated points for local search
        nondom_indices.sort(key=lambda i: np.sum(F[i]))
        k_parents = min(8, len(nondom_indices))
        parents = [X[i] for i in nondom_indices[:k_parents]]
        
        # Use remaining budget for local search with scipy
        remaining_budget = self.budget - n_evals
        if remaining_budget > 0 and parents:
            # Distribute evaluations among parents
            # Give each parent roughly equal share
            evals_per_parent = max(1, remaining_budget // len(parents))
            
            for parent in parents:
                if time.time() > deadline or n_evals >= self.budget:
                    break
                
                # Create a scalar objective using weighted sum
                # Use random weights to explore different parts of the Pareto front
                weights = np.random.dirichlet(np.ones(self.n_obj))
                
                def scalar_obj(x):
                    fx = problem(x)
                    n_evals += 1
                    return np.dot(weights, fx)
                
                # Use Nelder-Mead for derivative-free optimization
                try:
                    result = minimize(
                        scalar_obj, 
                        parent, 
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': evals_per_parent, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    # Record the result
                    fx_final = problem(result.x)
                    n_evals += 1
                    X = np.vstack([X, result.x])
                    F = np.vstack([F, fx_final])
                except Exception:
                    pass

        return np.array(X)
