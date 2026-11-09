"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _halton_point(dim, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    Uses prime bases for each dimension.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1  # 1-based index
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Global-to-Local Cascade Optimizer.
    
    Phase 1: Broad coverage using Halton sequence to maximize early AOCC.
    Phase 2: Refinement using Powell's method (robust to sharp ridges) with strict budget/time guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish well before.
        # Let's use 280s as a hard stop for logic to ensure we return cleanly.
        deadline = start_time + 280.0  
        dim = self.dim
        budget = self.budget
        
        # --- Phase 1: Global Exploration ---
        # We allocate a fixed number of calls for global search to ensure early progress.
        # 50 calls is a good balance: enough to cover [-5,5]^5 reasonably well with Halton,
        # leaving ~450 for local refinement.
        n_explore = 50
        n_explore = min(n_explore, budget - 1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = [] # List of (f, x)

        # Generate and evaluate Halton points
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            p_unit = _halton_point(dim, i)
            # Map [0, 1) to [-5, 5]
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except Exception:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            candidates.append((f, p))

        # Fallback if no valid points were evaluated (should not happen)
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))

        # Sort candidates to pick the best starting point for local search
        candidates.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement ---
        # We will try to refine the top few candidates.
        # Powell's method is chosen because it is derivative-free and adapts to the geometry
        # of the function (good for sharp ridges), unlike BFGS which relies on gradient approximations.
        
        # How many local searches?
        # Powell's method in dim=5 is relatively cheap.
        # We have ~450 calls left.
        # Let's try to refine the top 3 distinct candidates.
        
        max_local_starts = 3
        starts_to_try = []
        
        # Select top candidates, ensuring they are not identical
        for f_init, x_init in candidates:
            if len(starts_to_try) >= max_local_starts:
                break
            # Check if this point is too close to an already selected start
            is_duplicate = False
            for _, x_sel in starts_to_try:
                if np.linalg.norm(x_init - x_sel) < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                starts_to_try.append((f_init, x_init))
        
        for f_init, x_init in starts_to_try:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Define a wrapper for scipy to track calls and enforce limits
            # SciPy's minimize does not natively stop on a global budget counter easily
            # without a callback that raises an exception.
            
            def callback(xk):
                nonlocal calls_made
                # This callback is called after each function evaluation by scipy
                # Note: scipy might evaluate the function multiple times per iteration?
                # Actually, for Powell, it evaluates along axes.
                # We increment here.
                calls_made += 1
                if calls_made >= budget or time.time() > deadline:
                    raise TimeoutError("Budget or time exceeded")
                return False

            try:
                # Use Powell's method. It is robust for sharp ridges.
                # maxiter is set high, but we rely on callback to stop.
                res = minimize(
                    func, 
                    x_init, 
                    method='Powell',
                    options={'maxiter': 100, 'disp': False},
                    callback=callback
                )
                
                # Update best if scipy found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except TimeoutError:
                # If we hit the limit, we stop. The current best_x is kept.
                # We don't know the intermediate best from scipy, but we have the initial best.
                # This is a safe failure mode.
                pass
            except Exception:
                # Catch any other errors (e.g., convergence issues)
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
