"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import itertools

def fit(X_train, y_train):
    """
    Symbolic regression via evolutionary search over a rich function library,
    with a fallback to interpolation if the search fails to find a good fit.
    """
    start_time = time.time()
    DEADLINE = 300  # seconds total budget
    SAFETY_MARGIN = 10  # stop this far before deadline
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Ensure 1D input for simplicity (all equations have 1 input)
    if d == 1:
        x = X.flatten()
    else:
        x = X  # handle multi-dim if needed, but task says 1 input
    
    # Sort by x for interpolation fallback
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Build a rich feature library for 1D input
    # We'll use a set of basis functions and try to find a linear combination
    # or a small symbolic expression
    
    # Approach: Use a genetic algorithm to search over symbolic expressions
    # with a bounded set of operations and constants.
    
    # Define the building blocks
    def make_features(x):
        """Generate a rich set of features from x."""
        feats = []
        # Polynomial terms
        for deg in range(1, 8):
            feats.append(x ** deg)
        # Trigonometric
        feats.append(np.sin(x))
        feats.append(np.cos(x))
        feats.append(np.sin(2 * x))
        feats.append(np.cos(2 * x))
        feats.append(np.sin(0.5 * x))
        feats.append(np.cos(0.5 * x))
        # Exponential
        feats.append(np.exp(x))
        feats.append(np.exp(-x))
        feats.append(np.exp(0.5 * x))
        feats.append(np.exp(-0.5 * x))
        # Rational
        feats.append(1.0 / (x + 0.1))
        feats.append(1.0 / (x - 0.1))
        feats.append(1.0 / (x ** 2 + 1))
        # Logarithmic (with safe handling)
        abs_x = np.abs(x) + 1e-10
        feats.append(np.log(abs_x))
        feats.append(np.log1p(abs_x))
        # Square root
        feats.append(np.sqrt(abs_x))
        # Sigmoid-like
        feats.append(1.0 / (1.0 + np.exp(-x)))
        feats.append(1.0 / (1.0 + np.exp(x)))
        
        return np.column_stack(feats)
    
    # First, try a simple linear regression on the rich feature set
    F = make_features(x)
    
    # Use ridge regression for stability
    from numpy.linalg import lstsq
    
    # Try different regularization strengths
    best_w = None
    best_mse = np.inf
    
    for alpha in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]:
        try:
            # Ridge regression: (F^T F + alpha I)^{-1} F^T y
            if alpha > 0:
                A = F.T @ F + alpha * np.eye(F.shape[1])
                b = F.T @ y
                w = np.linalg.solve(A, b)
            else:
                w, *_ = lstsq(F, y, rcond=None)
            
            pred = F @ w
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w.copy()
        except:
            continue
    
    # Check if we got a very good fit
    y_var = np.var(y)
    if y_var > 1e-12:
        normalized_mse = best_mse / y_var
    else:
        normalized_mse = best_mse
    
    # If the rich feature linear fit is excellent, return it
    if normalized_mse < 1e-6:
        def predict_rich(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            F_test = make_features(x_test)
            return F_test @ best_w
        return predict_rich
    
    # Try a symbolic regression approach with a small expression tree
    # We'll search for expressions of the form: a * f(b * x + c) + d
    # or a * f(x) + b * g(x) + c
    
    # Define candidate functions
    candidate_funcs = [
        lambda t: t,
        lambda t: t ** 2,
        lambda t: t ** 3,
        lambda t: np.sin(t),
        lambda t: np.cos(t),
        lambda t: np.exp(t),
        lambda t: np.exp(-t),
        lambda t: 1.0 / (1.0 + np.exp(-t)),
        lambda t: np.sqrt(np.abs(t)),
        lambda t: np.log(np.abs(t) + 1e-10),
        lambda t: np.tanh(t),
        lambda t: np.sin(t) + np.cos(t),
        lambda t: np.exp(-t ** 2),
        lambda t: t * np.exp(-t ** 2),
    ]
    
    best_expr_mse = np.inf
    best_expr_pred = None
    best_expr_params = None
    
    # Search over pairs of functions and their parameters
    # Expression: a * f1(b * x + c) + d * f2(e * x + f) + g
    
    def evaluate_expr(x, f1, f2, params):
        """Evaluate a two-term expression."""
        a, b, c, d, e, f, g = params
        try:
            val = a * f1(b * x + c) + d * f2(e * x + f) + g
            if not np.all(np.isfinite(val)):
                return np.inf
            return np.mean((val - y) ** 2)
        except:
            return np.inf
    
    # Use differential evolution to optimize parameters
    # But first, try a grid search over function pairs
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback to interpolation
        pass  # will handle below
    
    # Try each pair of functions
    n_funcs = len(candidate_funcs)
    pairs_tried = 0
    
    for i in range(n_funcs):
        for j in range(i, n_funcs):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - SAFETY_MARGIN:
                break
            
            f1 = candidate_funcs[i]
            f2 = candidate_funcs[j]
            
            # Objective function for DE
            def obj(params):
                return evaluate_expr(x, f1, f2, params)
            
            # Bounds for parameters
            # a, d in [-10, 10], b, e in [-2, 2], c, f in [-5, 5], g in [-10, 10]
            bounds = [(-10, 10), (-2, 2), (-5, 5), (-10, 10), (-2, 2), (-5, 5), (-10, 10)]
            
            try:
                result = differential_evolution(obj, bounds, maxiter=50, popsize=15, 
                                                 tol=1e-12, seed=42, polish=False)
                pairs_tried += 1
                
                if result.fun < best_expr_mse:
                    best_expr_mse = result.fun
                    best_expr_params = result.x
                    best_expr_pred = (f1, f2)
            except:
                continue
    
    if best_expr_mse < best_mse:
        best_mse = best_expr_mse
        best_expr = best_expr_pred
        best_expr_params = best_expr_params
        
        if y_var > 1e-12:
            normalized_mse = best_mse / y_var
        else:
            normalized_mse = best_mse
        
        if normalized_mse < 1e-6:
            f1, f2 = best_expr
            a, b, c, d, e, f, g = best_expr_params
            
            def predict_expr(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test.flatten()
                val = a * f1(b * x_test + c) + d * f2(e * x_test + f) + g
                # Clip to finite values
                val = np.nan_to_num(val, nan=0.0, posinf=1e10, neginf=-1e10)
                return val
            return predict_expr
    
    # Fallback: use the best of (rich feature linear, expression)
    if best_mse < np.inf:
        if normalized_mse < 1e-6:
            def predict_rich(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test.flatten()
                F_test = make_features(x_test)
                pred = F_test @ best_w
                return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return predict_rich
    
    # Final fallback: interpolation
    # Use a cubic spline or linear interpolation
    try:
        # Remove duplicates if any
        unique_idx = np.concatenate(([True], np.diff(x_sorted) > 1e-12))
        x_unique = x_sorted[unique_idx]
        y_unique = y_sorted[unique_idx]
        
        if len(x_unique) >= 4:
            spline = interp1d(x_unique, y_unique, kind='cubic', fill_value='extrapolate')
            def predict_spline(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test.flatten()
                pred = spline(x_test)
                return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return predict_spline
        else:
            interp = interp1d(x_unique, y_unique, kind='linear', fill_value='extrapolate')
            def predict_linear(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(-1, 1)
                x_test = X_test.flatten()
                pred = interp(x_test)
                return np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            return predict_linear
    except:
        # Last resort: return mean
        y_mean = np.mean(y)
        def predict_mean(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            return np.full(X_test.shape[0], y_mean)
        return predict_mean

# EVOLVE-BLOCK-END
