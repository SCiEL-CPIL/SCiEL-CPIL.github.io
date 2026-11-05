"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_value = 2.0
    
    def evaluate(h):
        h = np.array(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Golay complementary pair approach:
    # h = a + b where a and b are optimized sequences
    # Complementary property: autoconv(a) + autoconv(b) has only DC term nonzero
    # We parameterize a and b as smooth functions with few parameters
    
    def make_h_from_params(params):
        """
        params: [A1, f1, phi1, A2, f2, phi2, B]
        h = B + A1*cos(2*pi*f1*t + phi1) + A2*cos(2*pi*f2*t + phi2)
        This creates a pair of cosines that can approximate complementary behavior
        """
        A1, f1, phi1, A2, f2, phi2, B = params
        i = np.arange(N)
        t = i / N
        h = B + A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)
        return h

    def objective(params):
        h = make_h_from_params(params)
        return evaluate(h)

    # Bounds for parameters
    bounds = [
        (-5.0, 5.0),    # A1
        (0.5, 20.0),    # f1
        (0.0, 2 * np.pi), # phi1
        (-5.0, 5.0),    # A2
        (0.5, 20.0),    # f2
        (0.0, 2 * np.pi), # phi2
        (0.5, 5.0)      # B (bias to keep sum nonzero)
    ]
    
    # Multi-restart differential evolution with more aggressive settings
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        try:
            seed_val = 42 + restart * 137
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=60, 
                popsize=10, 
                tol=0.001,
                seed=seed_val,
                init='sobol'
            )
            
            if res.fun < best_value:
                best_h = make_h_from_params(res.x)
                best_value = res.fun
                
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 20:
        return best_h

    # Local refinement with Nelder-Mead on the best found
    if best_value < 2.0:
        try:
            # Use the last successful res.x if available
            if 'res' in locals() and res.success:
                res_nm = minimize(
                    objective, 
                    res.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < best_value:
                    best_h = make_h_from_params(res_nm.x)
                    best_value = res_nm.fun
        except Exception:
            pass
    
    # Additional local search: try perturbing the best solution
    if time.time() < deadline - 10:
        try:
            # Generate random perturbations around best_h
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                noise = np.random.randn(N) * 0.01
                h_perturbed = best_h + noise
                val = evaluate(h_perturbed)
                if val < best_value:
                    best_h = h_perturbed
                    best_value = val
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline:
        return best_h

    return best_h
# EVOLVE-BLOCK-END
