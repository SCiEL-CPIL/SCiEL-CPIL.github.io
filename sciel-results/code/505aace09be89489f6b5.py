import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Search ---
        # BBOB-21 (Gallagher 101 peaks) has many local optima.
        # We need a good initial spread to find the basin of the global optimum.
        # The current program uses 40% LHS. Inspiration 1 suggests 60%.
        # Let's stick to a robust 40-50% allocation but use a more efficient sampling if possible.
        # LHS is good. Let's use 40% (200 calls) for global search.
        
        n_global = int(budget * 0.4)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            try:
                f_val = func(lhs[i])
                f_values[i] = f_val
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = lhs[i].copy()
            except Exception:
                # If func fails, we must be careful. The contract says exception later keeps calls made.
                # But we should probably stop or handle gracefully.
                break

        if calls == 0:
            return best_x, 0.0

        # Sort by fitness to get top candidates
        sorted_indices = np.argsort(f_values)
        
        # --- Phase 2: Local Refinement ---
        # Inspiration 2 suggests Nelder-Mead. Inspiration 1 suggests (1+1)-ES.
        # The "Lessons" mention Powell method improved speed and score (0.39 -> 1.27).
        # Powell is often faster and more robust for this type of smooth-ish but multimodal landscape
        # when started from a good point, especially in low dimensions (dim=5).
        # Let's use Powell for the top candidates.
        
        n_local_candidates = 5
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        # Powell's method in scipy.optimize.minimize might not strictly respect a "call budget" in the same way
        # as a manual loop, but we can monitor time and calls.
        # However, scipy's Powell can be expensive in function calls.
        # A safer approach for strict budget adherence is a custom (1+1)-ES or Nelder-Mead with call counting.
        # But the lesson says Powell worked well. Let's try to use scipy.optimize.minimize with a wrapper
        # that counts calls and stops if we exceed the per-candidate budget or global deadline.
        
        budget_per_local = max(10, remaining_budget // n_local_candidates)
        
        # We will try the top 5 candidates from the global search.
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = lhs[sorted_indices[k]].copy()
            f0 = f_values[sorted_indices[k]]
            
            # Skip if this point is very far from the best? 
            # In BBOB-21, local optima can be deep. Let's try them all.
            
            # Wrapper to count calls and enforce budget
            local_calls = 0
            max_local_calls = budget_per_local
            
            def local_func(x):
                nonlocal calls, local_calls, best_f, best_x
                if calls >= budget or local_calls >= max_local_calls or time.time() > deadline:
                    raise TimeoutError("Budget exceeded")
                calls += 1
                local_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                # Use Powell method
                # maxiter is a rough proxy, but we rely on the call counter inside local_func
                res = minimize(local_func, x0, method='Powell', options={'maxiter': 50})
            except TimeoutError:
                pass
            except Exception:
                # Catch any other exceptions from the optimizer or func
                pass

            # If Powell didn't improve or finished, we could try another method, 
            # but let's keep it simple and stick to the top structural change: Powell.
            # If we have remaining budget, we could run a second pass with (1+1)-ES on the best found so far.
            # But for now, let's see if Powell alone on top 5 is sufficient.
            
        return best_x, best_f
