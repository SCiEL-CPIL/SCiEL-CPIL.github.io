"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for robust initial exploration.
    - Phase 2: Multi-start Powell's method for robust valley navigation.
    - Phase 3: L-BFGS-B polishing for high-precision convergence.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit (evaluator kills at 330s)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        # Use enough points to ensure we hit the valley structure
        n_initial = min(64, self.budget // 5)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0
        
        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Powell's Method
        # Powell's method is derivative-free and robust for low-dimensional valley structures
        # It avoids the gradient estimation issues of L-BFGS in narrow valleys
        n_restarts = min(10, len(evaluations))
        bounds = [(-5, 5)] * self.dim
        
        # We allocate remaining budget to these restarts
        # Powell can be expensive, so we limit iterations
        # 500 budget - 64 initial = 436 remaining. 
        # 10 restarts * ~20-40 calls each = 200-400 calls.
        
        for i in range(n_restarts):
            if time.time() > deadline - 15:
                break
            if calls_used >= self.budget - 20:
                break
                
            fx0, x0 = evaluations[i]
            
            def powell_func(x):
                return safe_call(x)

            try:
                # Powell's method is good for the global structure
                res = minimize(
                    powell_func,
                    x0,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'xtol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            # If Powell found a better point, it's already in best_x/best_f via safe_call
            # But let's explicitly update if res.fun is better (it should be, but just in case)
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)
                evaluations.append((best_f, best_x.copy()))
                evaluations.sort(key=lambda t: t[0])

        # Phase 3: L-BFGS-B Polishing
        # Use the best point found by Powell (or Sobol) to do a high-precision gradient-based finish
        if calls_used < self.budget - 10 and time.time() < deadline - 10:
            if best_x is not None:
                def lbfgs_func(x):
                    return safe_call(x)

                try:
                    res = minimize(
                        lbfgs_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'gtol': 1e-15,
                            'ftol': 1e-15,
                        }
                    )
                except Exception:
                    pass
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)

        # Phase 4: If budget remains, try a few more random restarts with Powell
        if calls_used < self.budget - 30 and time.time() < deadline - 20:
            possible_restarts = (self.budget - calls_used) // 40
            for _ in range(possible_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_used >= self.budget - 10:
                    break
                
                # Sample near the current best but with some perturbation to escape local minima
                if best_x is not None:
                    x0 = best_x + rng.normal(0, 0.5, self.dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                
                def powell_func2(x):
                    return safe_call(x)

                try:
                    res = minimize(
                        powell_func2,
                        x0,
                        method='Powell',
                        bounds=bounds,
                        options={
                            'maxiter': 30,
                            'ftol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
