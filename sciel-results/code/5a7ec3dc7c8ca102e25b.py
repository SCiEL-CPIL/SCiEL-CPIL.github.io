import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination.
    Uses the zeros of the Laguerre polynomial L_12 as the initial guess,
    then performs a local optimization to minimize the variance of the roots,
    which serves as a proxy for minimizing the spatial spread/uncertainty.
    """
    start_time = time.time()
    deadline = 600.0 - 10.0  # 590s safety margin

    # 1. Get the 12 positive roots of the Laguerre polynomial L_12(x)
    # These are the canonical positive roots associated with this construction.
    try:
        x, _ = roots_laguerre(12)
        x = np.sort(x)
    except Exception:
        # Fallback to evenly spaced positive numbers if SciPy fails
        x = np.linspace(0.5, 12.0, 12)

    # 2. Define an objective function to minimize.
    # Hypothesis: Minimizing the spread (variance) of the roots reduces the 
    # effective bandwidth or support, potentially lowering the uncertainty bound.
    def objective(roots):
        # Ensure positivity is handled by bounds, but add penalty just in case
        if np.any(roots <= 0):
            return 1e10
        # Minimize the variance of the roots
        return np.var(roots)
    
    # 3. Optimization
    # Use L-BFGS-B with bounds to ensure roots remain positive.
    # Initial guess: Laguerre zeros
    # Bounds: [1e-6, None] for each root
    bounds = [(1e-6, None) for _ in range(12)]
    
    try:
        if time.time() < deadline:
            res = minimize(objective, x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success:
                x = res.x
            else:
                # If optimization fails or times out, stick with Laguerre zeros
                pass
    except Exception:
        pass

    # 4. Final sanity check
    x = np.sort(x)
    if np.any(x <= 0):
        # Force positivity if numerical issues occurred
        x = np.abs(x) + 1e-6
        x = np.sort(x)

    return [float(r) for r in x]
