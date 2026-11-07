"""Symbolic regression for a nonlinear oscillator: find a closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135  # 15s buffer before 150s kill

def _time_left() -> float:
    return _DEADLINE - time.time()


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Args:
        x: position array
        v: velocity array  
        params: 10-element array of fitted constants
    
    Returns:
        Array of acceleration predictions (one per row)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base linear terms
    result = p0 * x + p1 * v + p2
    
    # Nonlinear position terms
    result += p3 * x**2
    result += p4 * x**3
    
    # Nonlinear velocity terms
    result += p5 * v**2
    
    # Cross terms
    result += p6 * x * v
    result += p7 * x**2 * v
    
    # Higher order
    result += p8 * v**3
    result += p9 * x**2 * v**2
    
    return result


def _mse_loss(params, x, v, y):
    """Compute MSE for given parameters."""
    try:
        pred = equation(x, v, params)
        if pred.shape != y.shape:
            return 1e30
        if not np.all(np.isfinite(pred)):
            return 1e30
        return np.mean((pred - y)**2)
    except Exception:
        return 1e30


def _fit_and_evaluate(x, v, y):
    """Fit parameters using BFGS and return the fitted equation function."""
    # Start from ones as specified
    x0 = np.ones(10)
    
    try:
        result = minimize(_mse_loss, x0, args=(x, v, y), method='BFGS', 
                         options={'maxiter': 500, 'gtol': 1e-8})
        if result.success or result.fun < 1e29:
            return result.x
    except Exception:
        pass
    
    # Fallback: return ones if fitting fails
    return x0


def _evaluate_score(x, v, y, params):
    """Compute normalized MSE score."""
    try:
        pred = equation(x, v, params)
        if pred.shape != y.shape:
            return 1e30
        if not np.all(np.isfinite(pred)):
            return 1e30
        var_y = np.var(y)
        if var_y < 1e-12:
            return 1e30
        return np.mean((pred - y)**2) / var_y
    except Exception:
        return 1e30


def _main(x, v, y):
    """Main fitting routine with time guard."""
    # Initial fit
    params = _fit_and_evaluate(x, v, y)
    best_score = _evaluate_score(x, v, y, params)
    best_params = params.copy()
    
    # Try multiple random restarts if time allows
    rng = np.random.RandomState(42)
    for i in range(20):
        if _time_left() < 5:
            break
        
        # Random perturbation around current best
        scale = 1.0 / (1 + i * 0.1)
        perturb = rng.normal(0, scale, size=10)
        x0 = best_params + perturb
        
        try:
            result = minimize(_mse_loss, x0, args=(x, v, y), method='BFGS',
                             options={'maxiter': 200, 'gtol': 1e-8})
            if result.fun < best_score * np.var(y):
                score = _evaluate_score(x, v, y, result.x)
                if score < best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            continue
    
    return best_params


# The equation function is the main entry point as required
# The fitting is done by the evaluator, so we just need to provide equation()
# But we should ensure it works correctly with the evaluator's fitting procedure
# EVOLVE-BLOCK-END
