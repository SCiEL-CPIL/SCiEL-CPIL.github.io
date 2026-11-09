import numpy as np
import time
import cma
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf

    def _record(self, x, fx):
        self._eval_count += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = np.array(x)

    def _make_func(self, func):
        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = float(func(x))
            self._record(x, fx)
            return fx
        return wrapped

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 300.0
        
        wfunc = self._make_func(func)
        
        # Initial evaluation at origin
        x0 = np.zeros(self.dim)
        f0 = float(wfunc(x0))
        
        if self._eval_count >= self.budget:
            return self._best_x, self._best_f

        bounds = [(-5.0, 5.0)] * self.dim
        
        # CMA-ES
        popsize = 15
        # Reserve some budget for polish, e.g., 50
        max_cma_evals = self.budget - 50
        max_iter = max(1, max_cma_evals // popsize)
        
        try:
            es = cma.CMAEvolutionStrategy(
                x0, 
                2.0, 
                bounds=bounds,
                seed=42,
                popsize=popsize,
                maxiter=max_iter
            )
            
            while not es.stop() and self._eval_count < self.budget:
                if (time.time() - self._start_time) > deadline:
                    break
                    
                xs = es.ask()
                fs = []
                for x in xs:
                    if self._eval_count >= self.budget:
                        break
                    fx = float(wfunc(x))
                    fs.append(fx)
                
                if len(fs) < len(xs):
                    break
                
                es.tell(xs, fs)
                
        except Exception:
            pass

        # Polish
        remaining = self.budget - self._eval_count
        if remaining > 10 and self._best_x is not None and (time.time() - self._start_time) < deadline:
            try:
                # Use Nelder-Mead for local refinement
                # maxiter is roughly the number of function calls
                res = minimize(
                    wfunc,
                    x0=self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass
        
        return self._best_x, self._best_f
