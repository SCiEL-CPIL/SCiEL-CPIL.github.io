# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for WFG7 using weighted-sum scalarization
    with diverse weight vectors to encourage spread across the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Track all evaluated points and their objective values
        X_all = []
        F_all = []
        eval_count = [0]

        def evaluate(x):
            """Evaluate a point, record it, and return objectives."""
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).flatten()
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count[0] += 1
            return f

        def time_ok():
            return time.time() < deadline and eval_count[0] < self.budget

        # Generate diverse weight vectors for weighted-sum scalarization
        # Use simplex lattice for uniform coverage of the weight space
        n_weights = 30  # number of weight vectors
        weights = []
        
        # Add extreme points (each objective minimized alone)
        for i in range(m):
            w = np.zeros(m)
            w[i] = 1.0
            weights.append(w)
        
        # Add uniform random weights on the simplex
        for _ in range(n_weights - m):
            w = np.random.random(m)
            w /= np.sum(w)
            weights.append(w)
        
        # Add some intermediate weights for better coverage
        for i in range(m):
            for j in range(m):
                if i != j:
                    w = np.zeros(m)
                    w[i] = 0.5
                    w[j] = 0.5
                    weights.append(w)
        
        # Shuffle weights for diversity
        np.random.shuffle(weights)

        # Phase 1: Initial random sampling for diversity
        n_init = min(20, self.budget // 5)
        for i in range(n_init):
            if not time_ok():
                break
            x = lo + (hi - lo) * np.random.random(n)
            evaluate(x)

        # Phase 2: Weighted-sum scalarization with L-BFGS-B
        for w in weights:
            if not time_ok():
                break
            
            # Create scalarized objective function
            def scalarized_obj(x, w=w):
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float).flatten()
                return np.dot(w, f)
            
            # Create gradient approximation using finite differences
            def scalarized_grad(x, w=w):
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float).flatten()
                grad = np.zeros(n)
                eps = 1e-6
                for i in range(n):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_plus = np.clip(x_plus, lo, hi)
                    f_plus = np.asarray(problem(x_plus), dtype=float).flatten()
                    grad[i] = (np.dot(w, f_plus) - np.dot(w, f)) / eps
                return grad

            # Use a random starting point
            x0 = lo + (hi - lo) * np.random.random(n)
            
            # Optimize with L-BFGS-B
            try:
                result = minimize(
                    scalarized_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(bounds),
                    options={'maxiter': 50, 'maxfun': 10}
                )
                
                # Evaluate the result
                if time_ok():
                    x_opt = result.x
                    f_opt = evaluate(x_opt)
            except Exception:
                continue

        return np.array(X_all) if X_all else np.zeros((1, n))
# EVOLVE-BLOCK-END
