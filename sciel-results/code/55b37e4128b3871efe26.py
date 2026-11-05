"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the evaluator's value: abs(2 * N * max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    # Full autoconvolution
    ac = np.convolve(h, h, mode='full')
    peak = np.max(ac)
    return abs(2 * N * peak / (s ** 2))

def _objective(coeffs):
    """Objective function for differential_evolution: minimize the value."""
    # Reconstruct h from Chebyshev coefficients
    # Use x in [-1, 1] for Chebyshev evaluation
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    for i, c in enumerate(coeffs):
        h += c * np.polynomial.chebyshev.chebval(x, [0]*i + [1])
    # Ensure sum is not zero and finite
    if not np.all(np.isfinite(h)):
        return 1e10
    if np.sum(h) == 0:
        return 1e10
    return _value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try multiple random restarts with different Chebyshev degrees
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Degrees to try: low-degree Chebyshev polynomials
    degrees = [3, 5, 7, 9]
    
    for deg in degrees:
        if time.time() > deadline - 30:  # Need 30s safety margin
            break
            
        # Initial bounds for coefficients
        # Center coefficients around zero, scale by 1
        bounds = [(-1, 1)] * (deg + 1)
        
        # Differential evolution settings
        try:
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            
            if res.success and res.fun < best_val:
                # Reconstruct h
                x = np.linspace(-1, 1, N)
                h = np.zeros(N)
                for i, c in enumerate(res.x):
                    h += c * np.polynomial.chebyshev.chebval(x, [0]*i + [1])
                
                if np.all(np.isfinite(h)) and np.sum(h) != 0:
                    best_h = h
                    best_val = res.fun
                    
        except Exception as e:
            # If DE fails, try a simpler random search
            pass
        
        # Also try a few random coefficient sets for this degree
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            coeffs = np.random.uniform(-1, 1, deg + 1)
            x = np.linspace(-1, 1, N)
            h = np.zeros(N)
            for i, c in enumerate(coeffs):
                h += c * np.polynomial.chebyshev.chebval(x, [0]*i + [1])
            
            if np.all(np.isfinite(h)) and np.sum(h) != 0:
                val = _value(h)
                if val < best_val:
                    best_h = h
                    best_val = val
    
    return best_h
# EVOLVE-BLOCK-END
