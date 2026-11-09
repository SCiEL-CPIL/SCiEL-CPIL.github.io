import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        
        calls_used = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = None
        dim = self.dim
        
        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = [halton(i, base) for base in range(2, dim + 2)]
                points.append(point)
            return np.array(points)

        # Phase 1: Deterministic Space-Filling Exploration (Halton Sequence)
        # We use a fixed number of Halton points to ensure uniform coverage.
        # For 5D, 50 points is a reasonable start to find basins early.
        n_halton = 50
        halton_points = halton_sequence(n_halton, dim)
        
        # Map [0,1]^d to [-5,5]^d
        halton_points = halton_points * 10.0 - 5.0
        
        for x in halton_points:
            if time.time() > deadline or calls_used >= max_calls:
                break
            f_val = func(x)
            calls_used += 1
            update_best(x, f_val)

        # Phase 2: Local Refinement using Nelder-Mead on the best found points
        # We will refine the top few candidates found in Phase 1.
        # Since we only tracked the single best, we might miss other good basins.
        # To be more robust, we should keep track of a few distinct good points.
        # However, the current structure only tracks the single best.
        # Let's refine the single best point first, then if budget remains,
        # we can do a second round of Halton (shifted) and refine again,
        # or simply continue Nelder-Mead with restarts if it fails.
        
        # For simplicity and robustness in this step, we refine the current best.
        # Nelder-Mead is deterministic and fast for low dimensions.
        
        # We need a wrapper for scipy that respects the budget and time limit.
        # SciPy's minimize doesn't support callbacks for budget checking easily in all versions,
        # so we will implement a manual Nelder-Mead or use scipy with a tight maxiter.
        # Given the 300s limit and 500 calls, scipy is safe if maxiter is small.
        
        # Let's use scipy.optimize.minimize with method='Nelder-Mead'
        # We will run it on the best_x found so far.
        
        # To handle the budget, we can't easily interrupt scipy.
        # So we must estimate the cost. Nelder-Mead in 5D takes ~dim+1 initial points + iterations.
        # If we have used 50 calls, we have 450 left.
        # A single NM run might take 50-200 calls.
        # We can afford a few restarts.
        
        # Strategy:
        # 1. Run NM on best_x.
        # 2. If it improves, update best.
        # 3. If budget remains, generate a new random point (or next Halton point) and run NM on it.
        # 4. Repeat until budget/time exhausted.
        
        # To make this more effective, let's keep a list of "promising" points.
        # Since we only stored the single best in Phase 1, let's re-evaluate a few
        # random points to find other basins if the first NM doesn't get us close to 0 (or low).
        
        # Actually, a better strategy for multimodal:
        # After Phase 1, we have a best_x.
        # Run NM on best_x.
        # Then, generate new starting points (random or Halton continuation) and run NM on them.
        
        # Let's implement a helper to run NM with a call limit.
        # Since we can't easily limit calls inside scipy, we will use a wrapper
        # that counts calls and raises an exception or returns early if limit hit.
        # However, scipy might not handle exceptions well mid-optimization.
        
        # Alternative: Implement a simple (1+1) ES or Nelder-Mead manually?
        # The prompt suggests using the Halton + NM strategy from Inspiration 1.
        # Let's try to use scipy but with a tight maxiter and check time between runs.
        
        # We will define a function that calls func and increments the counter.
        # We will use scipy.minimize.
        
        def nm_func(x):
            nonlocal calls_used
            if calls_used >= max_calls:
                return best_f # Return current best to stop progress? No, just return a value.
            f_val = func(x)
            calls_used += 1
            return f_val

        # We need to be careful: if calls_used hits max, we should stop.
        # SciPy will keep calling. We can't stop it externally easily.
        # So we must ensure we have enough budget for one NM run.
        # If remaining budget < 20, skip NM.
        
        # Let's do a loop of NM restarts.
        # Start points:
        # 1. The best from Halton.
        # 2. Random points.
        
        start_points = [best_x.copy()]
        
        # Generate a few random start points to add to the queue
        rng = np.random.default_rng(42)
        for _ in range(10):
            if calls_used + 20 > max_calls: # Need ~20 calls for NM
                break
            start_points.append(rng.uniform(-5, 5, size=dim))
            
        # Shuffle start points? No, best first is good.
        
        for sp in start_points:
            if time.time() > deadline or calls_used >= max_calls - 10:
                break
            
            # Run Nelder-Mead
            # maxiter: limit iterations to prevent runaway.
            # In 5D, NM takes ~ (dim+1) * iterations.
            # Let's set maxiter=100. Worst case ~500 calls? Too high.
            # Let's set maxiter=50. ~250 calls worst case.
            # We check time before starting.
            
            try:
                res = minimize(nm_func, sp, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, do a final random sweep?
        # The Halton + NM strategy should be sufficient.
        
        return best_x, best_f
