"""Bootstrap portfolio: search for a better closed-form equation for the nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 120  # 30s buffer


def _time_left() -> float:
    return _DEADLINE - time.time()


def _is_valid(pred: np.ndarray, target: np.ndarray) -> bool:
    """Check if prediction is valid (correct shape, finite values)."""
    if pred.shape != target.shape:
        return False
    if not np.all(np.isfinite(pred)):
        return False
    return True


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute MSE between prediction and target."""
    if not _is_valid(pred, target):
        return 1e30
    diff = pred - target
    return float(np.mean(diff * diff))


def _make_candidate(basis_func):
    """Create a candidate equation function from a basis function."""
    def candidate(x, v, params):
        # basis_func returns an array of shape (n, 10)
        B = basis_func(x, v)
        return B @ params
    return candidate


# Define various basis functions for the nonlinear oscillator
# Each basis function takes (x, v) and returns an array of shape (n, 10)

def _basis_linear(x, v):
    """Linear basis: x, v, constant."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    return B


def _basis_poly2(x, v):
    """Polynomial basis up to degree 2."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x * x
    B[:, 3] = x * v
    B[:, 4] = v * v
    B[:, 5] = x * x * x
    B[:, 6] = x * v * v
    B[:, 7] = x * x * v
    B[:, 8] = v * v * v
    B[:, 9] = x * x * x * x
    return B


def _basis_poly3(x, v):
    """Polynomial basis up to degree 3."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x * x
    B[:, 3] = x * v
    B[:, 4] = v * v
    B[:, 5] = x * x * x
    B[:, 6] = x * v * v
    B[:, 7] = x * x * v
    B[:, 8] = x * x * x * x
    B[:, 9] = v * v * v
    return B


def _basis_trig(x, v):
    """Trigonometric and polynomial mix."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = np.sin(x)
    B[:, 3] = np.cos(x)
    B[:, 4] = x * x
    B[:, 5] = v * v
    B[:, 6] = x * v
    B[:, 7] = np.sin(x) * x
    B[:, 8] = np.cos(x) * v
    B[:, 9] = x * x * v
    return B


def _basis_mixed(x, v):
    """Mixed nonlinear basis: common oscillator terms."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x * x
    B[:, 3] = x * x * x
    B[:, 4] = v * v
    B[:, 5] = x * v
    B[:, 6] = x * x * v
    B[:, 7] = v * v * x
    B[:, 8] = np.tanh(x)
    B[:, 9] = np.tanh(v)
    return B


def _basis_softplus(x, v):
    """Softplus and polynomial mix."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x * x
    B[:, 3] = v * v
    B[:, 4] = x * v
    B[:, 5] = np.log(1.0 + np.exp(x))
    B[:, 6] = np.log(1.0 + np.exp(-x))
    B[:, 7] = np.log(1.0 + np.exp(v))
    B[:, 8] = np.log(1.0 + np.exp(-v))
    B[:, 9] = x * x * v
    return B


def _basis_rational(x, v):
    """Rational-like basis with denominators."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x / (1.0 + x * x)
    B[:, 3] = v / (1.0 + v * v)
    B[:, 4] = x * x
    B[:, 5] = v * v
    B[:, 6] = x * v
    B[:, 7] = x * x * x
    B[:, 8] = v * v * v
    B[:, 9] = x * x * v
    return B


def _basis_abs(x, v):
    """Absolute value and polynomial mix."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = np.abs(x)
    B[:, 3] = np.abs(v)
    B[:, 4] = x * x
    B[:, 5] = v * v
    B[:, 6] = x * v
    B[:, 7] = x * x * x
    B[:, 8] = np.abs(x * v)
    B[:, 9] = x * x * v
    return B


def _basis_exp(x, v):
    """Exponential and polynomial mix."""
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = np.exp(-x * x)
    B[:, 3] = np.exp(-v * v)
    B[:, 4] = x * x
    B[:, 5] = v * v
    B[:, 6] = x * v
    B[:, 7] = np.exp(-x * x) * x
    B[:, 8] = np.exp(-v * v) * v
    B[:, 9] = x * x * v
    return B


# List of candidate basis functions
_basis_funcs = [
    _basis_linear,
    _basis_poly2,
    _basis_poly3,
    _basis_trig,
    _basis_mixed,
    _basis_softplus,
    _basis_rational,
    _basis_abs,
    _basis_exp,
]


def _fit_candidate(basis_func, x_data, v_data, y_data):
    """Fit a candidate model and return (mse, best_params, equation_func)."""
    B = basis_func(x_data, v_data)
    
    def objective(params):
        pred = B @ params
        if not np.all(np.isfinite(pred)):
            return 1e30
        diff = pred - y_data
        return float(np.mean(diff * diff))
    
    # Start from ones as per the evaluator
    x0 = np.ones(10)
    
    try:
        result = minimize(objective, x0, method='BFGS', options={'maxiter': 5000})
        best_params = result.x
        best_mse = result.fun
        
        if not np.all(np.isfinite(best_params)) or not np.isfinite(best_mse):
            return 1e30, best_params, None
        
        # Verify the prediction is valid
        pred = B @ best_params
        if not _is_valid(pred, y_data):
            return 1e30, best_params, None
        
        # Create the equation function
        def equation_func(x, v, params):
            B_new = basis_func(x, v)
            return B_new @ params
        
        return best_mse, best_params, equation_func
    except Exception:
        return 1e30, x0, None


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Main entry point. The evaluator will call this with training data to fit params,
    then with test data to get predictions.
    
    However, looking at the task contract more carefully:
    "The evaluator fits params, an array of length 10, with scipy.optimize.minimize..."
    "equation must return an array with one prediction per row."
    
    So the evaluator calls equation(x, v, params) where params is already fitted.
    But wait - how does the evaluator fit params? It must be calling equation internally
    during the optimization. So equation needs to work with any params array of length 10.
    
    The strategy: we need to pick a good basis function that generalizes well.
    Since we can't see the data, we'll use a flexible nonlinear basis that captures
    common oscillator dynamics.
    """
    # Use a mixed basis that captures polynomial, trigonometric, and tanh nonlinearities
    n = len(x)
    B = np.ones((n, 10))
    B[:, 0] = x
    B[:, 1] = v
    B[:, 2] = x * x
    B[:, 3] = x * x * x
    B[:, 4] = v * v
    B[:, 5] = x * v
    B[:, 6] = x * x * v
    B[:, 7] = v * v * x
    B[:, 8] = np.tanh(x)
    B[:, 9] = np.tanh(v)
    return B @ params
# EVOLVE-BLOCK-END
