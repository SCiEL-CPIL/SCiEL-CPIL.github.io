import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Exploration
        # We want to find the global basin quickly.
        # The minimizer is in [-4, 4]^5.
        # Rosenbrock has a narrow valley. Random points might miss it.
        # Let's use a mix of:
        # 1. Origin
        # 2. Points along axes (to catch shifted centers)
        # 3. Random points in [-4, 4]^5
        # 4. Points near the "typical" Rosenbrock valley structure (shifted/rotated unknown)
        
        # Since rotation and shift are hidden, we must rely on exploration.
        # However, 500 calls is very small for 5D global optimization.
        # A good strategy is to run a few short L-BFGS-B runs from diverse starts.
        # L-BFGS-B is very efficient for Rosenbrock once in the basin.
        
        # Let's generate ~10 diverse starting points.
        starts = []
        starts.append(np.zeros(self.dim))
        
        # Axis-aligned points
        for i in range(self.dim):
            for val in [-2.0, 2.0]:
                p = np.zeros(self.dim)
                p[i] = val
                starts.append(p)
        
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4.0, 4.0, self.dim))
            
        # Shuffle to ensure we don't just try origin first every time (though rng is seeded)
        # Actually, we want to try the most promising ones first?
        # We don't know which is promising without evaluating.
        # Let's evaluate all starts first? 10 calls is cheap.
        
        evaluated_starts = []
        for x0 in starts:
            if call_count >= self.budget - 50:
                break
            fx = tracked_func(x0)
            evaluated_starts.append((fx, x0.copy()))
            
        # Sort by function value
        evaluated_starts.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement (Race)
        # We will try to polish the top candidates.
        # L-BFGS-B is great for Rosenbrock.
        # We allocate budget to the best starts.
        
        # Let's try the top 3 starts with L-BFGS-B.
        # If one converges well, we stop.
        
        # We need to be careful with budget.
        # 500 calls total.
        # 10 used for starts.
        # 490 left.
        
        # Strategy:
        # 1. Run L-BFGS-B from best start with ~100 calls.
        # 2. If it improves significantly, continue polishing with ~200 calls.
        # 3. If it stagnates, try second best start.
        
        # Let's implement a "polish" function that runs L-BFGS-B with a specific budget limit.
        
        def polish(x_start, budget_limit):
            nonlocal call_count, best_x, best_fx
            if call_count >= self.budget - 5 or not time_ok():
                return
            
            # Calculate how many calls we can afford
            available = min(budget_limit, self.budget - call_count - 5)
            if available < 5:
                return
            
            try:
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': available,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
            except Exception:
                pass
        
        # Try top 3 starts
        # We give the first one a larger initial budget to see if it's the global basin.
        # If it gets very low, we keep polishing it.
        
        for i in range(min(3, len(evaluated_starts))):
            if not time_ok() or call_count >= self.budget - 10:
                break
            
            fx_start, x_start = evaluated_starts[i]
            
            # If we already found a very good solution, skip worse starts
            if i > 0 and best_fx < 1e-3:
                break
            
            # Initial burst: 50 calls
            polish(x_start, 50)
            
            # Check improvement
            # If best_fx is very low, we are likely in the global basin.
            # Let's check if we can afford more polishing.
            
            # If we have used < 200 calls and best_fx is still not tiny, try the next start?
            # Or just keep polishing the best one?
            
            # For Rosenbrock, once in the valley, L-BFGS-B converges fast.
            # If it didn't converge in 50 calls, it might be stuck or just slow.
            # Let's give it another 100 calls if it's the best start.
            
            if i == 0:
                # Give the best start a second chance
                if call_count < self.budget - 50 and best_fx > 1e-6:
                    polish(best_x, 100)
            
            # If we have budget left and haven't converged, try next start
            if call_count < self.budget - 100 and best_fx > 1.0:
                continue
            else:
                # If we are doing well, break and do final polish
                break

        # Phase 3: Final Polish
        # Use remaining budget to polish the global best.
        if time_ok() and call_count < self.budget - 5:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                try:
                    minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
            
            # If still budget left, try Nelder-Mead for robustness
            if time_ok() and call_count < self.budget - 5:
                remaining = self.budget - call_count - 5
                if remaining > 10:
                    try:
                        minimize(
                            tracked_func,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 200,
                                'maxfev': remaining,
                                'adaptive': True,
                                'xatol': 1e-10,
                                'fatol': 1e-12
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_fx
