"""
Strategy: Fixed-Knot B-Spline Basis with Linear Coefficients.

Instead of a fixed linear combination of generic functions (polynomials, trig, etc.) which
leads to an ill-conditioned linear regression problem that BFGS struggles with, we use
a B-Spline basis. B-Splines are locally supported, which means the design matrix is sparse
and well-conditioned. This makes the problem effectively a linear least-squares problem
disguised as a nonlinear one, allowing BFGS to converge quickly to the global optimum
regardless of the starting point (all ones).

We use a moderate number of knots (e.g., 20-30) to ensure high expressiveness while keeping
the condition number manageable. The 10 parameters are the coefficients of 10 selected
B-Spline basis functions (or a subset of a larger set, but here we map 10 params to 10
specific basis functions).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.special import factorial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Deadline guard: 150s limit, keep safety margin of 5s
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # 5s safety margin

def safe_minimize(fun, x0, bounds=None):
    """Safe wrapper around scipy.optimize.minimize that returns None on failure."""
    try:
        result = minimize(fun, x0, method='BFGS', bounds=bounds, options={'maxiter': 200, 'gtol': 1e-5})
        if not np.all(np.isfinite(result.x)):
            return None
        return result
    except Exception:
        return None

def compute_mse(params, x_train, y_train):
    """Compute MSE with error handling for invalid predictions."""
    try:
        pred = equation(x_train, params)
        if not np.all(np.isfinite(pred)) or len(pred) != len(y_train):
            return 1e30
        mse = np.mean((pred - y_train) ** 2)
        if not np.isfinite(mse):
            return 1e30
        return mse
    except Exception:
        return 1e30

def fit_params(x_train, y_train, initial_guesses=None):
    """Fit parameters using BFGS, trying multiple initial guesses if needed."""
    n_params = 10
    
    if initial_guesses is None:
        initial_guesses = [np.ones(n_params)]
    
    best_params = None
    best_mse = np.inf
    
    for x0 in initial_guesses:
        if not np.all(np.isfinite(x0)):
            continue
        if len(x0) != n_params:
            continue
            
        def objective(params):
            return compute_mse(params, x_train, y_train)
        
        result = safe_minimize(objective, x0)
        if result is not None:
            mse = objective(result.x)
            if mse < best_mse:
                best_mse = mse
                best_params = result.x
    
    return best_params, best_mse

# Global cache for training data and best model
_train_data = None
_best_model = None
_best_score = np.inf
_has_trained = False

def _bspline_basis(x, knots, degree=3):
    """
    Compute B-Spline basis functions for given knots and degree.
    Returns an array of shape (len(x), len(knots)-degree-1)
    """
    # Clip x to the knot range to avoid issues
    x_min = knots[0]
    x_max = knots[-1]
    x_clipped = np.clip(x, x_min, x_max)
    
    n_basis = len(knots) - degree - 1
    n_points = len(x_clipped)
    
    # Build the basis matrix
    basis = np.zeros((n_points, n_basis))
    
    # Use De Boor's algorithm or recursive definition
    # For simplicity and robustness, we use a direct evaluation approach
    # This is a simplified version that works for uniform and non-uniform knots
    
    for i in range(n_basis):
        # B-Spline basis function i
        # Using the Cox-de Boor recursion
        # Base case: degree 0
        if degree == 0:
            mask = (x_clipped >= knots[i]) & (x_clipped < knots[i+1])
            basis[:, i] = mask.astype(float)
        else:
            # Recursive case
            term1 = 0.0
            term2 = 0.0
            
            # First term
            if knots[i+degree] - knots[i] > 1e-12:
                term1 = ((x_clipped - knots[i]) / (knots[i+degree] - knots[i])) * _bspline_basis(x_clipped, knots, degree-1)[:, i]
            
            # Second term
            if knots[i+degree+1] - knots[i+1] > 1e-12:
                term2 = ((knots[i+degree+1] - x_clipped) / (knots[i+degree+1] - knots[i+1])) * _bspline_basis(x_clipped, knots, degree-1)[:, i+1]
            
            basis[:, i] = term1 + term2
    
    return basis

def _get_basis(x, n_params=10):
    """
    Get the B-Spline basis matrix for the given x, selecting n_params basis functions.
    We use a set of knots that span the domain [0.1, 6].
    """
    # Define knots: include domain boundaries and interior knots
    # We want a total of n_params + degree + 1 knots
    # Let's use degree 3, so we need n_params + 4 knots
    # But we can also use a fixed set of knots and select a subset of basis functions
    
    # Let's use a fixed set of knots that gives us more than 10 basis functions,
    # then select the first 10 (or a well-distributed subset)
    degree = 3
    # Interior knots: evenly spaced in [0.1, 6]
    # Let's use 15 interior knots to get 18 basis functions
    interior_knots = np.linspace(0.5, 5.5, 15)
    knots = np.concatenate(([0.1, 0.1, 0.1, 0.1], interior_knots, [6.0, 6.0, 6.0, 6.0]))
    
    # Compute all basis functions
    all_basis = _bspline_basis(x, knots, degree)
    
    # Select 10 basis functions
    # We can select the first 10, or a well-distributed subset
    n_basis_available = all_basis.shape[1]
    if n_basis_available >= n_params:
        # Select evenly spaced basis functions
        indices = np.linspace(0, n_basis_available - 1, n_params, dtype=int)
        basis = all_basis[:, indices]
    else:
        # Pad with zeros if we have fewer basis functions
        basis = np.zeros((len(x), n_params))
        basis[:, :n_basis_available] = all_basis
    
    return basis

def train_on_data(x_train, y_train):
    """Train the model on the given data and return the best score achieved."""
    global _best_model, _best_score, _has_trained
    
    if _has_trained:
        return _best_score
    
    _train_data = (x_train, y_train)
    _has_trained = True
    
    # Compute variance of y for normalization
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_var = 1.0
    
    # Fit the B-Spline model
    # The equation function uses B-Spline basis, so we just need to fit the coefficients
    initial_guesses = [
        np.ones(10),
        np.zeros(10),
        np.linspace(0.1, 1.0, 10),
    ]
    
    p, mse = fit_params(x_train, y_train, initial_guesses=initial_guesses)
    if p is not None and mse < np.inf:
        pred = equation(x_train, p)
        if np.all(np.isfinite(pred)):
            norm_mse = mse / y_var
            _best_score = norm_mse
            _best_model = (p,)
    else:
        # Fallback
        _best_score = 1.0
        _best_model = (np.ones(10),)
    
    return _best_score

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Main equation function using B-Spline basis.
    params are the coefficients of the 10 selected B-Spline basis functions.
    """
    # Get the B-Spline basis
    basis = _get_basis(x, n_params=10)
    
    # Compute prediction as linear combination
    pred = basis @ params
    
    return pred

# EVOLVE-BLOCK-END
