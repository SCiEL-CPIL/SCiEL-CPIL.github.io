# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-objective Optimizer using LHS initialization and Weighted Sum Local Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        # 330s limit. Safety margin of 50s.
        self.deadline = self.start_time + 280

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                # Check for duplicates to save space/memory, though not strictly necessary for scoring
                x_clipped = np.clip(x, lo, hi)
                for i in range(len(X_list)):
                    if np.allclose(X_list[i], x_clipped, atol=1e-10):
                        return
                X_list.append(x_clipped)
                F_list.append(f)

        # Phase 1: Initialization using Latin Hypercube Sampling
        # 40 points for initialization.
        n_init = 40
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        samples = sampler.random(n=n_init)
        # Map to bounds
        samples = lo + samples * (hi - lo)
        
        for x in samples:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
                
        if not F_list:
            # Fallback if no evaluations were made
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None: add_point(x0, f0)
            return X_list if X_list else [x0]

        # Helper to get non-dominated points
        def get_nondominated_indices(F):
            n = len(F)
            if n == 0:
                return []
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if is_dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        # Phase 2: Weighted Sum Local Refinement
        # Strategy: Use specific weight vectors to target different parts of the Pareto front.
        # We use the "Chebyshev" set of weights which ensures coverage of the convex hull.
        # For 3 objectives, we use:
        # 1. (1, 0, 0) -> Minimize f0
        # 2. (0, 1, 0) -> Minimize f1
        # 3. (0, 0, 1) -> Minimize f2
        # 4. (1, 1, 1) -> Minimize sum (center)
        # 5. (1, 1, 0) -> Minimize f0+f1
        # 6. (1, 0, 1) -> Minimize f0+f2
        # 7. (0, 1, 1) -> Minimize f1+f2
        
        # Given budget 200, init 40, remaining 160.
        # 7 weight vectors. 160 / 7 ~= 22 evals each.
        # Let's use maxfev=25 for each to be safe, or adjust dynamically.
        
        weight_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0])
        ]
        
        # Normalize weights to sum to 1 for consistent scaling in objective
        weight_vectors = [w / np.sum(w) for w in weight_vectors]
        
        # Determine how many local searches we can afford
        # Each local search consumes ~maxfev evaluations.
        # We have 160 budget.
        # Let's allocate maxfev = 20 per search. 7 * 20 = 140. Leaves 20 buffer.
        maxfev_per_search = 20
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Identify non-dominated points to use as starting points
        nd_indices = get_nondominated_indices(F)
        if len(nd_indices) == 0:
             sums = np.sum(F, axis=1)
             nd_indices = np.argsort(sums)[:len(weight_vectors)]
        
        # If we have fewer non-dominated points than weight vectors, cycle through them
        # or pick the best ones.
        
        for i, w in enumerate(weight_vectors):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            # Select a starting point. 
            # Ideally, start from the point that is currently best for this weight direction.
            # Score = dot(w, F)
            scores = np.dot(F, w)
            # Find the index in the full list with the best score
            best_idx = np.argmin(scores)
            
            x_start = X[best_idx].copy()
            
            def objective(x, w=w):
                f = self._evaluate(x, problem)
                if f is None:
                    return 1e6
                # Weighted sum
                return np.dot(w, f)
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={
                        'maxfev': maxfev_per_search, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.x is not None:
                    f_final = self._evaluate(res.x, problem)
                    if f_final is not None:
                        add_point(res.x, f_final)
                        
            except Exception:
                pass

        return X_list
# EVOLVE-BLOCK-END
