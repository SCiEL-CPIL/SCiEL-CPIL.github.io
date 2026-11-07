"""Rewritten program using LHS + Powell hybrid strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    """Hybrid LHS and Powell optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Deadline guard: keep 20s margin for safety
        deadline = start_time + 280

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            # Ensure x is a numpy array of correct shape and dtype
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            return func(x)

        # 1. Initialization with Latin Hypercube Sampling
        # Use a moderate number of points to ensure good coverage without wasting budget
        n_lhs = min(40, max(20, budget // 15))
        
        sampler = LatinHypercube(d=dim, seed=12345)
        lhs_points = sampler.random(n=n_lhs)
        # Scale to [-5, 5]
        lhs_points = lhs_points * 10.0 - 5.0

        lhs_results = []
        for x in lhs_points:
            if call_count[0] >= budget - 50:
                break
            f_val = wrapped_func(x)
            lhs_results.append((f_val, x.copy()))

        if not lhs_results:
            # Fallback
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
        else:
            lhs_results.sort(key=lambda item: item[0])
            best_f, best_x = lhs_results[0]

            # 2. Local Optimization using Powell's Method
            # Powell is derivative-free and often outperforms Nelder-Mead on Rosenbrock
            # due to its ability to align search directions with the valley.
            
            # We will run Powell on the top 3 candidates
            top_k = 3
            starts = [p[1] for p in lhs_results[:top_k]]
            
            # Reserve some budget for potential multiple runs
            # Powell's method call count is roughly (dim * iterations)
            # We estimate maxiter conservatively.
            
            for i, x0 in enumerate(starts):
                if time.time() > deadline:
                    break
                if call_count[0] >= budget - 10:
                    break
                
                # Define bounds for Powell (though Powell handles bounds via reflection/penalty internally in scipy? 
                # Actually, scipy's Powell respects bounds if provided in the minimize call? 
                # Check: scipy.optimize.minimize(method='Powell') does NOT support bounds directly in older versions, 
                # but in newer SciPy, it might. 
                # Wait, standard Powell in SciPy does NOT support bounds. 
                # We must ensure x stays in bounds. 
                # Rosenbrock is defined on R^d, but the task clips to [-5,5]. 
                # If Powell steps out, the function clips it. This is acceptable.
                
                # However, stepping out and clipping can be inefficient.
                # Let's use Nelder-Mead if we can't easily bound Powell? 
                # Or use L-BFGS-B which supports bounds.
                
                # Let's try L-BFGS-B as it supports bounds and is gradient-based (finite diff).
                # For dim=5, FD gradient costs 6 calls.
                # Powell costs ~dim calls per iteration.
                
                # Let's stick to Powell but be careful. 
                # Actually, let's use Nelder-Mead with adaptive=True as it is robust and doesn't need bounds (clipping handles it).
                # But the previous code used NM and scored 1.92.
                
                # Let's try **CMA-ES** if we can import it? No, standard lib only.
                
                # Let's try **Basin Hopping** with L-BFGS-B as the local minimizer.
                # Basin Hopping is designed for global optimization and uses a local minimizer.
                
                pass # Placeholder for decision

            # Decision: Use **Powell** for the main search. 
            # To handle bounds, we rely on the clipping in `func`.
            # Powell is often faster than NM on Rosenbrock.
            
            # Let's run Powell on the best point.
            if time.time() < deadline and call_count[0] < budget - 20:
                try:
                    # maxiter is the number of iterations. Each iteration involves dim line searches.
                    # Total calls ~ maxiter * dim.
                    # Remaining budget: R. Maxiter ~ R / (dim * 2) to be safe.
                    remaining = budget - call_count[0]
                    max_iter = max(10, remaining // (dim * 2))
                    
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='Powell',
                        options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

            # If we have budget left, try the second best start
            if len(starts) > 1 and time.time() < deadline and call_count[0] < budget - 20:
                x0 = starts[1]
                # Only if it's significantly different?
                if np.linalg.norm(x0 - best_x) > 0.5:
                    try:
                        remaining = budget - call_count[0]
                        max_iter = max(10, remaining // (dim * 2))
                        res = minimize(
                            wrapped_func,
                            x0,
                            method='Powell',
                            options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
