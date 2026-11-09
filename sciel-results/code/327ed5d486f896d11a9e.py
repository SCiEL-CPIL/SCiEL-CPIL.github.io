import numpy as np
import time
from scipy.stats.qmc import Sobol

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin within 300s limit
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Wrapper to track calls, time, and best result
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= self.budget:
                raise Exception("Budget exceeded")
            
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Phase 1: Global Exploration with Sobol Sequence
        n_global = min(100, self.budget)
        
        try:
            sampler = Sobol(d=self.dim, scramble=True)
            sobol_points = sampler.random(n_global)
            sobol_points = sobol_points * 10.0 - 5.0
            
            for x in sobol_points:
                wrapped_func(x)
                
        except (TimeoutError, Exception):
            pass

        if best_x is None:
            x0 = np.zeros(self.dim)
            try:
                wrapped_func(x0)
            except:
                pass
            if best_x is None:
                return np.zeros(self.dim), np.inf

        # Phase 2: Local Refinement with CMA-ES
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 10:
            try:
                from cma import CMA
                sigma0 = 0.5
                opts = {
                    'maxfevals': remaining_budget,
                    'seed': 42,
                    'bounds': [-5.0, 5.0],
                    'verbose': -9,
                    'tolx': 1e-10,
                    'tolopt': 1e-10,
                    'tolstagnation': 50,
                    'CMA_essfun': 200
                }
                
                es = CMA(best_x.copy(), sigma0, opts)
                
                while not es.stop() and calls_made < self.budget and time.time() < deadline:
                    batch = es.ask()
                    values = []
                    for x in batch:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        val = wrapped_func(x)
                        values.append(val)
                    
                    if len(values) == len(batch):
                        es.tell(batch, values)
                    
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                        
            except ImportError:
                pass
            except Exception:
                pass

        # Phase 3: Coordinate Descent Polish (better for ill-conditioned problems like bent cigar)
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 5 and best_x is not None:
            try:
                current_x = best_x.copy()
                current_f = best_f
                step_size = 0.5
                
                while remaining_budget > 0 and time.time() < deadline:
                    improved = False
                    for d in range(self.dim):
                        # Try both directions
                        for direction in [1, -1]:
                            trial_x = current_x.copy()
                            trial_x[d] += direction * step_size
                            trial_x = np.clip(trial_x, -5.0, 5.0)
                            
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            
                            trial_f = wrapped_func(trial_x)
                            remaining_budget = self.budget - calls_made
                            
                            if trial_f < current_f:
                                current_x = trial_x
                                current_f = trial_f
                                improved = True
                            else:
                                # If no improvement, reduce step size for this dimension
                                step_size *= 0.5
                                
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                    
                    if not improved:
                        # No improvement in full cycle, try smaller steps
                        step_size *= 0.5
                        if step_size < 1e-6:
                            break
                            
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
