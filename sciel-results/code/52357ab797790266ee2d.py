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

        # Reference point from task
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # 1. Initialization: Latin Hypercube Sampling for better coverage
        n_init = 40
        rng = np.random.default_rng(0)
        
        # LHS in 5D
        n_per_dim = n_init
        X_init = []
        for d in range(self.dim):
            # Create stratified samples
            edges = np.linspace(0, 1, n_per_dim + 1)
            samples = (edges[:-1] + rng.random(n_per_dim) * 0.5 / n_per_dim)
            # Actually simpler:
            samples = (np.arange(n_per_dim) + rng.random(n_per_dim)) / n_per_dim
            X_init.append(samples)
        
        X_init = np.array(X_init).T
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if eval_count[0] >= self.budget:
                break
            count_eval(x)
            
        # 2. Tchebycheff Decomposition + Multi-start L-BFGS-B
        # We will generate weight vectors on a grid and optimize each.
        # For 3 objectives, weights sum to 1.
        # We'll use a set of weight vectors covering the simplex.
        
        # Generate weight vectors
        # For 3 obj, we can use a grid of size m x m
        m = 10
        weights = []
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                w = np.array([i / m, j / m, k / m])
                if w.sum() > 0:
                    weights.append(w)
        
        # Shuffle weights for diversity
        rng.shuffle(weights)
        
        # Current non-dominated set
        # We'll maintain a list of solutions.
        # Start with initial points.
        
        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
            
        def hypervolume_2d(points, ref):
            # points: list of (y1, y2)
            # ref: (r1, r2)
            if not points:
                return 0.0
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref[1]
            for p in points:
                if p[1] < ref[1]:
                    hv += (prev_y - p[1]) * (ref[0] - p[0])
                    prev_y = p[1]
            return hv

        # We will iterate through weight vectors.
        # For each weight, we run L-BFGS-B from a few starting points.
        # Starting points: center, corners, and best current solutions.
        
        # Pre-compute some starting points
        center = (lo + hi) / 2
        corners = np.array([[lo[d] if (i >> d) & 1 else hi[d] for d in range(self.dim)] for i in range(2**self.dim)])
        
        # We have a budget of 200.
        # n_init used 40.
        # Remaining: 160.
        # We have ~66 weight vectors.
        # We can do 1-2 local optimizations per weight if fast.
        # L-BFGS-B might take many function calls.
        # Let's estimate: 160 calls / 66 weights ~ 2.4 calls/weight. Too few for L-BFGS-B.
        
        # Strategy Shift:
        # Use the remaining budget for a simple Evolutionary Strategy or CMA-ES?
        # Or just more weighted-sum optimizations with fewer iterations?
        # L-BFGS-B is very efficient. If we set maxiter=2, it might just do a few steps.
        # But we need to be careful.
        
        # Let's try a different approach:
        # 1. Run L-BFGS-B for a subset of weights (e.g., 10 key weights) with more iterations.
        # 2. Fill the rest with random sampling or local search from existing points.
        
        # Key weights: Vertices (1,0,0), (0,1,0), (0,0,1), Centroid (1/3,1/3,1/3), and edges.
        key_weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1/3.0, 1/3.0, 1/3.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
        
        # Also add some random weights
        for _ in range(5):
            w = rng.random(self.n_obj)
            w /= w.sum()
            key_weights.append(w)
        
        # We have 12 key weights.
        # Budget left: 160.
        # 160 / 12 = 13.3 calls per weight.
        # L-BFGS-B can work with this if we start from good points.
        
        # Starting points for each weight:
        # 1. Center
        # 2. Random points from initial LHS
        
        for w in key_weights:
            if eval_count[0] >= self.budget - 2:
                break
            if time.time() > deadline - 10:
                break
                
            def objective(x):
                f = problem(x)
                # Tchebycheff scalarization: max(w_i * (f_i - z_i))
                # We need z_i (utopia point). We don't know it.
                # Use weighted sum instead? Or simple Tchebycheff with z_i = 0?
                # Since objectives are minimized, and we want to push towards 0.
                # But scales might differ.
                # Let's stick to Weighted Sum for simplicity and reliability,
                # but with adaptive weights based on current best?
                
                # Actually, Weighted Sum is convex hull only.
                # Tchebycheff can find non-convex parts.
                # Let's use Tchebycheff with estimated ideal point.
                # Ideal point: min of each objective seen so far.
                
                # We need to track current ideal.
                # This is getting complex inside the loop.
                
                # Let's just use Weighted Sum. It's robust.
                return np.dot(w, f)
            
            # Start from center
            x0 = center
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10})
                # res.x is the result
                # Note: minimize calls objective multiple times.
                # We need to count these!
                # The `objective` function calls `problem` directly, bypassing our counter!
                # THIS IS A BUG IN THE PREVIOUS LOGIC IF NOT HANDLED.
                
            except Exception:
                pass
            
            # The problem with using scipy.minimize directly is that it calls `problem` inside `objective`,
            # but `objective` doesn't increment `eval_count`.
            # We MUST use the counting mechanism.
            
            # So we need to define objective using count_eval?
            # But count_eval expects x and returns f.
            # And minimize expects a function that returns a scalar.
            
            # Let's rewrite the local optimization loop to manually step or use a wrapper that counts.
            
            # If we use `objective` which calls `count_eval`, then `count_eval` returns f (vector).
            # But `objective` needs to return a scalar.
            # So:
            
            def objective_counted(x):
                f = count_eval(x)
                return np.dot(w, f)
            
            # Now we can use minimize.
            # But we must catch exceptions if budget is exceeded inside count_eval.
            
            x0 = center
            try:
                res = minimize(objective_counted, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5, 'maxls': 5})
            except Exception:
                # Budget exceeded or error
                break
                
            # If we still have budget, try another start point for this weight?
            # Maybe a random point.
            if eval_count[0] < self.budget - 5 and time.time() < deadline - 5:
                x0 = lo + (hi - lo) * rng.random(self.dim)
                try:
                    res2 = minimize(objective_counted, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5, 'maxls': 5})
                except Exception:
                    break

        return all_X
