"""Multi-Objective Optimizer for RE21 using Pareto-Gradient Descent (MO-LBFGS)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization: Latin Hypercube Sampling
        # Use a significant portion of the budget for initial exploration to ensure the front is well-covered.
        # 40 points is a good balance for dim=4.
        n_init = 40
        X_init = self._latin_hypercube(n_init, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        X_eval = X_init
        F_eval = F_init
        n_evals = len(X_eval)
        
        # 2. Pareto-Gradient Descent Loop
        # Strategy:
        # - Maintain a set of non-dominated points.
        # - Generate weights based on the current front (uniform or adaptive).
        # - For each weight, run a local optimization (L-BFGS-B) starting from multiple random points.
        # - Add new non-dominated points to the archive.
        
        # We will iterate until budget or time runs out.
        # Each "iteration" consists of a batch of local optimizations.
        
        # To be efficient, we'll group local optimizations.
        # Let's say we do 5 weight vectors at a time, each with 5 restarts.
        # That's 25 problem evaluations per batch (plus the local opt calls which are fast? 
        # Wait, problem(x) is the expensive call. L-BFGS-B calls problem multiple times.
        # So we must be careful with the budget.
        
        # Actually, L-BFGS-B uses the function value. Each call to minimize consumes budget.
        # A typical L-BFGS-B run might take 10-50 function evaluations.
        # If we do 25 runs, that's 250-1250 evals. We only have 200 total.
        # So we cannot do many full L-BFGS-B runs.
        
        # Revised Strategy:
        # 1. Init: 40 evals.
        # 2. Remaining: 160 evals.
        # 3. We can afford roughly 160 / 15 (avg evals per local opt) = ~10 local optimizations.
        # 4. To maximize hypervolume, we should spread these optimizations across the front.
        
        # Let's run a loop where we pick a weight vector, run a local opt from a random start, 
        # and check time/budget.
        
        # We need to track the non-dominated front to update weights.
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Get current non-dominated front
            nd_indices = self._get_non_dominated_indices(F_eval)
            if len(nd_indices) == 0:
                # Fallback if something went wrong
                break
                
            nd_points = F_eval[nd_indices]
            
            # Generate weight vectors
            # Simple strategy: Uniform weights on the simplex, but biased towards the current front's spread.
            # For 2 objectives, weights are (w1, 1-w1).
            # We can sample weights from a Beta distribution or just uniform.
            # To explore the whole front, uniform is good.
            # Let's pick a random weight vector.
            w1 = np.random.uniform(0.1, 0.9)  # Avoid extreme corners initially
            w = np.array([w1, 1 - w1])
            
            # Normalize weights to ensure they are on the simplex
            w = w / np.sum(w)
            
            # Define scalar objective
            # We want to minimize w1*f1 + w2*f2
            # But we should shift and scale to avoid one objective dominating due to scale.
            # Let's normalize objectives based on current min/max.
            f_min = np.min(F_eval, axis=0)
            f_max = np.max(F_eval, axis=0)
            f_range = f_max - f_min + 1e-8
            
            def scalar_obj(x):
                fx = problem(x)
                n_evals_local[0] += 1 # This won't work directly in closure easily without nonlocal or list
                # Actually, we need to count evaluations globally.
                # It's better to wrap problem to count.
                return np.dot(w, (fx - f_min) / f_range)
            
            # We need a counter for evaluations inside the local optimization
            # Let's create a wrapper
            eval_counter = [0]
            def counted_problem(x):
                eval_counter[0] += 1
                return problem(x)
            
            # Check budget before starting
            if n_evals + 10 > self.budget: # Reserve 10 for a quick opt
                break
                
            # Run local optimization
            # Start from a random point
            x0 = lo + np.random.random(self.dim) * span
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    lambda x: np.dot(w, (counted_problem(x) - f_min) / f_range),
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 20, 'maxfun': 20} # Limit function calls per run
                )
            except Exception:
                # If optimization fails, just use the starting point or a random point
                x_new = x0
                f_new = problem(x_new)
                eval_counter[0] += 1
            else:
                x_new = res.x
                # The last call in minimize might have been the best, but we need to ensure we record it.
                # L-BFGS-B returns the best x. We need to evaluate it one more time to be sure?
                # No, the function was called during minimize. But we need the value.
                # We can just call problem(x_new) again? That wastes 1 eval.
                # Better: The last call in minimize is likely x_new. But to be safe and simple:
                # We will just call problem(x_new) to get the final value and record it.
                # This adds 1 eval.
                f_new = problem(x_new)
                eval_counter[0] += 1
            
            # Update global counters
            n_evals += eval_counter[0]
            
            # Add to archive
            X_eval = np.vstack([X_eval, x_new])
            F_eval = np.vstack([F_eval, f_new])
            
            # Optional: If we have budget left, we could do more restarts for this weight.
            # But for simplicity and to spread out, let's stick to 1 restart per weight vector per loop.
            # The loop will pick a new weight next time.

        return X_eval

    def _latin_hypercube(self, n, dim, lo, hi):
        pts = np.zeros((n, dim))
        for i in range(dim):
            intervals = np.linspace(0, 1, n + 1)
            # Shuffle to ensure independence
            order = np.random.permutation(n)
            pts[:, i] = np.array([np.random.uniform(intervals[j], intervals[j+1]) for j in order])
        return lo + pts * (hi - lo)

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if not is_nd[j]:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_nd[i] = False
                    break
        return np.where(is_nd)[0]

# EVOLVE-BLOCK-END
