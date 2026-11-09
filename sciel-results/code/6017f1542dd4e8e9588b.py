"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Optimizer for Rosenbrock (f8) with hidden shifts.
    Strategy:
    1. Global exploration with diverse random and deterministic points.
    2. Local refinement using L-BFGS-B on the top candidates.
    L-BFGS-B is preferred over Nelder-Mead for Rosenbrock due to its 
    quasi-Newton properties which handle the narrow valley curvature better.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 290s
        self.deadline = self.start_time + 290.0
        n = self.dim
        total_budget = self.budget

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.asarray(x, dtype=float)
            if x.shape != (n,):
                x = x.reshape(n)
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if not np.isfinite(val):
                return 1e10

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # We want to find a point in the basin of attraction of the global minimum.
        # The global minimum of Rosenbrock is at (1, ..., 1).
        # With hidden shift s, min is at 1+s. s is in [-4, 4].
        # So min is in [-3, 5].
        
        # Allocate 30% of budget to exploration, min 50, max 150
        n_explore = max(50, min(150, int(total_budget * 0.3)))
        
        # Generate diverse starting points
        starts = []
        
        # 1. Deterministic points likely to be close or cover space
        starts.append(np.ones(n)) # Standard min
        starts.append(np.zeros(n)) # Center
        starts.append(np.full(n, 3.0)) # Upper mid
        starts.append(np.full(n, -3.0)) # Lower mid
        starts.append(np.full(n, 5.0)) # Upper bound
        starts.append(np.full(n, -5.0)) # Lower bound
        
        # 2. Random points in the likely region [-3, 5]
        # Since min is in [-3, 5], sampling here is more efficient
        while len(starts) < n_explore:
            # Mix of uniform in likely region and full box
            if rng.random() < 0.7:
                starts.append(rng.uniform(-3, 5, size=n))
            else:
                starts.append(rng.uniform(-5, 5, size=n))

        # Evaluate all exploration points
        history = []
        for x0 in starts:
            if self.calls_made >= total_budget or time.time() > self.deadline:
                break
            try:
                x0 = np.clip(x0, -5, 5)
                f0 = wrapped_func(x0)
                history.append((f0, x0.copy()))
            except StopIteration:
                break
            except Exception:
                continue

        if not history:
            # Fallback if something went wrong in exploration
            x0 = np.ones(n)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0
            except:
                self.best_x = np.ones(n)
                self.best_f = np.inf
            return self.best_x, self.best_f

        # Sort by function value
        history.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Take the top K candidates.
        # K=3 is a good balance. If the first one converges, we are done.
        # If not, we try the next.
        top_k = min(5, len(history))
        
        # We will iterate through the top candidates.
        # To avoid re-optimizing the same point, we track visited starts.
        # However, L-BFGS-B is deterministic for a given start, so if it failed 
        # to converge well the first time, it will likely fail again unless 
        # we perturb it. But since we are picking distinct top-k from exploration,
        # they are likely distinct.
        
        for i in range(top_k):
            if self.calls_made >= total_budget or time.time() > self.deadline:
                break
            
            x_start = history[i][1]
            
            # Skip if this start is very close to the current best (within 1e-6)
            # to avoid redundant work, though L-BFGS-B is cheap if already converged.
            if self.best_x is not None and np.linalg.norm(x_start - self.best_x) < 1e-6:
                continue

            remaining_budget = total_budget - self.calls_made
            if remaining_budget < 10:
                break
            
            try:
                # L-BFGS-B is robust and uses gradient info.
                # maxiter is limited by budget via wrapped_func.
                # ftol/gtol set tight to ensure high precision.
                lbfgs_res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 200, # High limit, but budget will stop it
                        'ftol': 1e-15,
                        'gtol': 1e-10,
                        'maxls': 30 # Limit line search steps
                    }
                )
                
                # wrapped_func updates self.best_f and self.best_x automatically
                # Check if we have effectively converged to a very low value
                # If best_f is very low, we can stop early to save time/budget
                # for other instances? No, budget is per instance.
                # But if we find a near-perfect minimum, we can stop.
                if self.best_f < 1e-5:
                    # Very good solution found, stop refining further
                    break
                    
            except StopIteration:
                break
            except Exception:
                # If L-BFGS-B fails, continue to next candidate
                continue

        # Final polish if we have budget left and haven't converged
        if self.best_x is not None and self.calls_made < total_budget and time.time() < self.deadline:
            remaining = total_budget - self.calls_made
            if remaining > 5:
                try:
                    final_res = scipy.optimize.minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': 50,
                            'ftol': 1e-18,
                            'gtol': 1e-12
                        }
                    )
                except StopIteration:
                    pass
                except Exception:
                    pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
