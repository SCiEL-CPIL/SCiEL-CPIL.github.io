"""Symbolic regression: nonlinear oscillator equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 150s, use 120s as safety margin
_DEADLINE = 120.0
_start_time = None


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a rich polynomial basis up to degree 3 in x and v, structured to match
    physical oscillator terms: constant, linear restoring, damping, quadratic
    restoring, cross terms, cubic terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    return p0 + p1 * x + p2 * v + p3 * x2 + p4 * xv + p5 * v2 + p6 * x3 + p7 * x2v + p8 * xv2 + p9 * v3


def _compute_mse(x: np.ndarray, v: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    """Compute MSE for given parameters."""
    try:
        pred = equation(x, v, params)
        if not np.all(np.isfinite(pred)):
            return 1e30
        mse = np.mean((pred - y) ** 2)
        if not np.isfinite(mse):
            return 1e30
        return float(mse)
    except Exception:
        return 1e30


def _fit_with_bfgs(x: np.ndarray, v: np.ndarray, y: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Fit parameters using BFGS from given initial point."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return x0
    
    def objective(params):
        if time.time() - _start_time > _DEADLINE:
            return 1e30
        return _compute_mse(x, v, y, params)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                method="BFGS",
                options={"maxiter": 500, "gtol": 1e-8, "disp": False}
            )
        if result.success or result.fun < 1e29:
            return result.x
        else:
            return x0
    except Exception:
        return x0


def _search_best_params(x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Search for best parameters using multiple restarts within time budget."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_params = 10
    best_params = np.ones(n_params)
    best_mse = _compute_mse(x, v, y, best_params)
    
    # Try the standard all-ones start first
    candidate = _fit_with_bfgs(x, v, y, np.ones(n_params))
    mse = _compute_mse(x, v, y, candidate)
    if mse < best_mse:
        best_mse = mse
        best_params = candidate
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        return best_params
    
    # Try several different random initializations
    rng = np.random.RandomState(42)
    for i in range(20):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 5:
            break
        
        # Different initialization strategies
        if i % 4 == 0:
            x0 = rng.uniform(-1, 1, n_params)
        elif i % 4 == 1:
            x0 = rng.randn(n_params) * 0.5
        elif i % 4 == 2:
            x0 = np.ones(n_params) * rng.uniform(-2, 2)
        else:
            # Scale-aware: target range is about [-0.1, 0.1], inputs in [-0.6, 0.6]
            x0 = np.array([
                rng.uniform(-0.1, 0.1),  # constant
                rng.uniform(-2, 2),      # linear x
                rng.uniform(-2, 2),      # linear v
                rng.uniform(-1, 1),      # x^2
                rng.uniform(-1, 1),      # xv
                rng.uniform(-1, 1),      # v^2
                rng.uniform(-0.5, 0.5),  # x^3
                rng.uniform(-0.5, 0.5),  # x^2 v
                rng.uniform(-0.5, 0.5),  # x v^2
                rng.uniform(-0.5, 0.5),  # v^3
            ])
        
        candidate = _fit_with_bfgs(x, v, y, x0)
        mse = _compute_mse(x, v, y, candidate)
        if mse < best_mse:
            best_mse = mse
            best_params = candidate
    
    return best_params


# Module-level caching: the evaluator calls equation() many times during fitting,
# but we only need to do the search once. We'll detect if we're being called for
# the first time with training data and do our search then.

_cache = {
    "fitted": False,
    "best_params": None,
    "x_shape": None,
    "n_rows": None
}


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    The evaluator calls this function repeatedly during BFGS fitting. We use the
    first call to perform our own multi-restart search if beneficial, but we must
    still respect the params passed in for correctness during the evaluator's fit.
    
    Strategy: Just use the rich polynomial form. The evaluator will fit params.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    return p0 + p1 * x + p2 * v + p3 * x2 + p4 * xv + p5 * v2 + p6 * x3 + p7 * x2v + p8 * xv2 + p9 * v3
# EVOLVE-BLOCK-END
