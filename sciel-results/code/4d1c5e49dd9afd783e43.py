"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    """CMA-ES based optimizer for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = np.random.uniform(-5, 5, dim)
        f_global_best = np.inf
        calls = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Sparse Random Sampling to identify promising basins
        # f22 is highly multimodal. We spend a small fraction of the budget
        # on random sampling to find the global minimum basin.
        n_init = 50
        for _ in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: CMA-ES on the best basin
        if HAS_CMA and check_time() and calls < budget:
            try:
                # Configure CMA-ES
                # Population size lambda
                # For dim=5, lambda = 4 + floor(3 * log(5)) = 4 + 5 = 9? 
                # Standard default is often 4 + 3*log(n). 
                # Let's use a slightly larger population for exploration in early phases of CMA-ES
                pop_size = 4 + int(3 * np.log(dim))
                if pop_size < 10:
                    pop_size = 10
                
                # Initial sigma: scale to the box size
                sigma0 = 2.0
                
                # Options
                opts = {
                    'maxiter': 1000, # Large limit, we control via budget
                    'seed': 42,
                    'verbose': -9, # No output
                    'tolx': 1e-8,
                    'tolfun': 1e-10,
                    'bounds': [-5, 5],
                    'popsize': pop_size
                }
                
                # Create problem
                # We need to wrap the function to count calls and respect time
                class CMAProblem:
                    def __init__(self, func, evaluate_fn, check_time_fn, budget, calls_ref):
                        self.func = func
                        self.evaluate_fn = evaluate_fn
                        self.check_time_fn = check_time_fn
                        self.budget = budget
                        self.calls_ref = calls_ref
                        self.best_x = None
                        self.best_f = np.inf

                    def __call__(self, x):
                        if not self.check_time_fn():
                            raise cma.StopIteration("Time limit reached")
                        if self.calls_ref[0] >= self.budget:
                            raise cma.StopIteration("Budget exceeded")
                        return self.evaluate_fn(x)

                    def result(self):
                        return self.best_x, self.best_f

                # cma.fmin uses its own random number generator internally unless specified
                # We pass the starting point as the global best from Phase 1
                x0 = x_global_best.copy()
                
                # We need to be careful with call counting in cma.fmin
                # cma.fmin returns (best_x, result, es)
                # We can't easily inject our 'evaluate' wrapper directly into cma.fmin's internal loop
                # without subclassing or using a custom evaluator if available.
                # However, standard cma.fmin calls the function.
                # Let's use a simple wrapper that increments our global counter.
                
                def wrapped_func(x):
                    if not check_time():
                        raise Exception("Time limit")
                    if calls >= budget:
                        raise Exception("Budget")
                    return evaluate(x)

                # Run CMA-ES
                try:
                    best_x, result, es = cma.fmin(wrapped_func, x0, sigma0, opts)
                    # Update global best if CMA found something better
                    if best_x is not None:
                        f_best = wrapped_func(best_x)
                        if f_best < f_global_best:
                            f_global_best = f_best
                            x_global_best = best_x.copy()
                except (cma.StopIteration, Exception) as e:
                    # If CMA stopped early or errored, we keep the best found so far
                    pass

            except Exception:
                pass
        else:
            # Fallback: Simple (1+1)-ES if cma is not available or failed
            if check_time() and calls < budget:
                x = x_global_best.copy()
                f = f_global_best
                sigma = 1.0
                C = np.eye(dim)
                
                while check_time() and calls < budget:
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    norm_z = np.linalg.norm(z)
                    if norm_z > 1e-10:
                        z = z / norm_z
                    
                    x_trial = x + sigma * z
                    f_trial = evaluate(x_trial)
                    
                    if f_trial < f:
                        C = (1 - 1.0/dim) * C + (1.0/dim) * (np.outer(z, z))
                        x = x_trial
                        f = f_trial
                    else:
                        # Shrink sigma occasionally
                        if rng.random() < 0.1:
                            sigma *= 0.9
                            C = 0.5 * C + 0.5 * np.eye(dim)

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
