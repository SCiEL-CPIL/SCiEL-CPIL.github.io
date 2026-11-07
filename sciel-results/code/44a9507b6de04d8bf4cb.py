"""Rewritten optimizer using Sobol sequence sampling + Powell local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import scipy.stats


class Optimizer:
    """Sobol global sampling followed by Powell local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = func(np.asarray(x, dtype=np.float64))
                call_count += 1
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                call_count += 1
                return np.inf

        best_x = None
        best_f = np.inf

        # Phase 1: Sobol Sequence Sampling
        # Use a low-discrepancy sequence to cover the space efficiently
        num_sobol = min(100, budget // 2)  # Reserve half budget for refinement
        if num_sobol < 10:
            num_sobol = 10
            
        results = []
        
        try:
            # Generate Sobol sequence
            sobol = scipy.stats.qmc.Sobol(d=d, scramble=True)
            # We need 2^k points for Sobol. Find appropriate k
            k = int(np.ceil(np.log2(num_sobol)))
            num_points = 2 ** k
            if num_points > budget:
                num_points = budget
                k = int(np.ceil(np.log2(num_points)))
                if 2**k > budget:
                    num_points = 2**(k-1)
            
            # Generate points in [0,1]^d
            points_unit = sobol.random(num_points)
            # Map to [-5, 5]^d
            points = points_unit * 10.0 - 5.0
            
            # Evaluate points
            for p in points:
                if call_count >= budget or time.time() > deadline:
                    break
                f = safe_call(p)
                if f < np.inf:
                    results.append((f, p))
                    if f < best_f:
                        best_f = f
                        best_x = p.copy()
                        
            # Sort results by fitness
            results.sort(key=lambda x: x[0])
            
        except Exception:
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(42)
            for _ in range(min(50, budget // 2)):
                if call_count >= budget or time.time() > deadline:
                    break
                p = rng.uniform(-5, 5, d)
                f = safe_call(p)
                if f < np.inf:
                    if f < best_f:
                        best_f = f
                        best_x = p.copy()
                    results.append((f, p))
            results.sort(key=lambda x: x[0])

        # Phase 2: Local Refinement with Powell
        # Take the top K best points from Phase 1
        top_k = min(5, len(results))
        
        if top_k > 0:
            # Define a wrapper for minimize that respects budget and time
            def func_wrapper(x):
                if call_count >= budget or time.time() > deadline:
                    return np.inf
                return safe_call(x)
            
            # Run Powell from each top point
            for i in range(top_k):
                if call_count >= budget or time.time() > deadline:
                    break
                
                x0 = results[i][1]
                
                # Use Powell method, derivative-free, good for ill-conditioned
                try:
                    res = scipy.optimize.minimize(
                        func_wrapper, 
                        x0, 
                        method='Powell', 
                        options={'maxiter': 50, 'xtol': 1e-6, 'ftol': 1e-6}
                    )
                    
                    # Check if result is better
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

        # Phase 3: If budget remains, do a few more random restarts + Powell
        if best_x is not None and call_count < budget:
            rng = np.random.default_rng(999)
            remaining_budget = budget - call_count
            num_restarts = min(5, remaining_budget // 20)
            
            for _ in range(num_restarts):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Perturb the current best or sample new
                if rng.random() < 0.5:
                    x0 = best_x + rng.normal(0, 0.5, d)
                else:
                    x0 = rng.uniform(-5, 5, d)
                x0 = np.clip(x0, -5, 5)
                
                f0 = safe_call(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                
                # Local refine
                try:
                    res = scipy.optimize.minimize(
                        func_wrapper, 
                        x0, 
                        method='Powell', 
                        options={'maxiter': 20, 'xtol': 1e-5, 'ftol': 1e-5}
                    )
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
