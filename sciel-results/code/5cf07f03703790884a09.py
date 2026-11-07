import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy:
        # 1. Moderate global sampling to establish baselines.
        # 2. Adaptive local refinement using L-BFGS-B (smooth) and Nelder-Mead (robust).
        # 3. Dynamic budget allocation for local runs to maximize the number of distinct basins explored.
        
        # Phase 1: Global Sampling
        # We sample enough points to likely hit multiple basins in 5D, but save the bulk for local search.
        # 40 samples is a good balance.
        num_global = 40
        candidates = []
        for i in range(num_global):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            candidates.append((f, x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # We iterate through the sorted candidates and run local optimization from each.
        # The key is dynamic budget allocation: give more budget to early runs if they are cheap,
        # or ensure we always have enough for several runs.
        
        # We target up to 15 local runs, but stop if budget/time runs out.
        max_local_runs = 15
        
        for i in range(max_local_runs):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Select start point
            if i < len(candidates):
                _, x_start = candidates[i]
            else:
                # If we exhausted our initial candidates, generate a new random one
                # This acts as a "random restart" for later iterations
                x_start = rng.uniform(-5.0, 5.0, d)
                # Evaluate it so it's in our record (though we don't add to candidates list here to keep it simple)
                call_func(x_start)
            
            # Dynamic maxfev calculation
            remaining = self.budget - calls_made
            runs_left = max_local_runs - i
            if runs_left <= 0:
                runs_left = 1
            
            # We want to reserve some budget for future runs.
            # Let's aim for an average of 15-20 calls per local run.
            # If remaining is large, we can afford more per run.
            # If remaining is small, we must be frugal.
            
            # Heuristic:
            # If we have > 100 calls left and < 5 runs left, we can be generous.
            # Otherwise, keep it tight.
            
            target_per_run = 20
            maxfev_lbfgs = int(min(remaining * 0.6, max(10, target_per_run * (remaining // (runs_left * 1.2)))))
            maxfev_lbfgs = max(10, min(50, maxfev_lbfgs))
            
            # Run L-BFGS-B
            # L-BFGS-B is very effective for Sphere, Rosenbrock, and smooth parts of Rastrigin/Gallagher
            res_x = x_start
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev_lbfgs,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                res_x = res.x
            except Exception:
                pass
            
            # Polish with Nelder-Mead
            # NM is robust to non-smoothness and can escape shallow local minima that L-BFGS-B might get stuck in
            # due to gradient noise or flat regions.
            if calls_made < self.budget and time.time() < deadline:
                # Allocate remaining budget for this iteration to NM
                remaining_iter = self.budget - calls_made
                # Give NM a smaller budget, e.g., 10-15 calls
                nm_maxfev = max(5, min(15, remaining_iter // max(1, runs_left)))
                
                try:
                    res_nm = minimize(
                        call_func,
                        res_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final Polish / Use remaining budget
        # If we have significant budget left (e.g. local optimizations converged very fast),
        # do a final intensive run from the global best.
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 20:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(50, remaining),
                            'ftol': 1e-15,
                            'gtol': 1e-10
                        }
                    )
                except Exception:
                    pass
                
                # One more NM polish if budget allows
                if calls_made < self.budget and time.time() < deadline:
                    remaining = self.budget - calls_made
                    if remaining > 5:
                        try:
                            minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': min(20, remaining),
                                    'xatol': 1e-10,
                                    'fatol': 1e-10
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            # Ensure at least one call is made if something went wrong
            call_func(best_x)

        return best_x, best_f
