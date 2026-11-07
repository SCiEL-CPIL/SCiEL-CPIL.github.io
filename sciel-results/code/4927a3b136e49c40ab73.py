"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE_MARGIN = 15.0  # seconds before the 330s wall-clock limit to stop
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def time_left():
        return (start_time + 330.0 - DEADLINE_MARGIN) - time.time()
    
    # Handle constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)

    # Sort training data by x for interpolation
    if d == 1:
        x = X[:, 0]
    else:
        x = X[:, 0]  # Use first dimension for 1D interpolation
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check if data is strictly monotonic in x (needed for PCHIP)
    # Remove duplicate x values by averaging y
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        # Average y values for duplicate x
        y_unique = np.array([y_sorted[unique_indices[i]:unique_indices[i+1]].mean() 
                            for i in range(len(unique_x))])
    else:
        y_unique = y_sorted
    
    # Create PCHIP interpolator
    try:
        pchip = PchipInterpolator(unique_x, y_unique)
        
        def predict_pchip(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            xp = X_pred[:, 0]
            return pchip(xp)
        
        # Check if PCHIP fits well on training data
        y_hat_train = pchip(unique_x)
        sse = np.sum((y_unique - y_hat_train)**2)
        sst = np.sum((y_unique - np.mean(y_unique))**2)
        r2 = 1 - sse / sst if sst > 1e-12 else 1.0
        
        # If PCHIP fits very well (R2 > 0.999), return it
        if r2 > 0.999:
            return predict_pchip
    except:
        pass
    
    # Strategy: Try symbolic regression with common function forms
    # Define a set of candidate functions with parameters
    def try_symbolic_fit():
        """Try to fit y = a*f(b*x+c) + d for various f"""
        best_r2 = -np.inf
        best_params = None
        best_func = None
        
        # Candidate base functions
        def make_funcs(x):
            funcs = []
            # Polynomial
            for deg in range(1, 6):
                funcs.append(('poly', deg, lambda x, d=deg: x**d))
            # Trig
            for k in [1, 2, 3]:
                funcs.append(('sin', k, lambda x, k=k: np.sin(k*x)))
                funcs.append(('cos', k, lambda x, k=k: np.cos(k*x)))
            # Exp
            funcs.append(('exp', 1, lambda x: np.exp(x)))
            funcs.append(('exp_neg', 1, lambda x: np.exp(-x)))
            # Log
            funcs.append(('log', 1, lambda x: np.log(np.abs(x) + 1e-10)))
            # Sqrt
            funcs.append(('sqrt', 1, lambda x: np.sqrt(np.abs(x) + 1e-10)))
            # Rational
            funcs.append(('inv', 1, lambda x: 1.0 / (x + 1e-10)))
            # Tanh
            funcs.append(('tanh', 1, lambda x: np.tanh(x)))
            return funcs
        
        # For 1D case, try fitting y = a * f(b*x + c) + d
        if d == 1:
            x_test = x_sorted
            y_test = y_unique
            
            for func_name, param, func in make_funcs(x_test):
                # Try different scales and shifts
                for b in [0.5, 1, 2, 3]:
                    for c in [-2, -1, 0, 1, 2]:
                        try:
                            f_vals = func(x_test * b + c)
                            f_vals = np.nan_to_num(f_vals, nan=0, posinf=0, neginf=0)
                            
                            # Fit y = a*f + d using least squares
                            A = np.column_stack([f_vals, np.ones_like(f_vals)])
                            w, *_ = np.linalg.lstsq(A, y_test, rcond=None)
                            a, d_val = w
                            
                            y_hat = A @ w
                            sse = np.sum((y_test - y_hat)**2)
                            sst = np.sum((y_test - np.mean(y_test))**2)
                            r2 = 1 - sse / sst if sst > 1e-12 else 1.0
                            
                            if r2 > best_r2:
                                best_r2 = r2
                                best_params = (func_name, b, c, a, d_val)
                                best_func = lambda xp, fn=func_name, bb=b, cc=c, aa=a, dd=d_val: \
                                    aa * (np.exp(xp*bb+cc) if fn=='exp' else 
                                         np.exp(-(xp*bb+cc)) if fn=='exp_neg' else
                                         np.sin((xp*bb+cc)) if fn=='sin' else
                                         np.cos((xp*bb+cc)) if fn=='cos' else
                                         np.log(np.abs(xp*bb+cc)+1e-10) if fn=='log' else
                                         np.sqrt(np.abs(xp*bb+cc)+1e-10) if fn=='sqrt' else
                                         1.0/(xp*bb+cc+1e-10) if fn=='inv' else
                                         np.tanh(xp*bb+cc) if fn=='tanh' else
                                         (xp*bb+cc)**param) + dd
                        except:
                            continue
                        if time_left() < 5.0:
                            return best_r2, best_params, best_func
                if time_left() < 5.0:
                    return best_r2, best_params, best_func
        else:
            # For multi-dim, just use PCHIP on first dimension as fallback
            return -np.inf, None, None
        
        return best_r2, best_params, best_func
    
    best_r2_sym, best_params_sym, best_func_sym = try_symbolic_fit()
    
    # If symbolic fit is excellent, use it
    if best_r2_sym > 0.999 and best_func_sym is not None:
        def predict_symbolic(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            xp = X_pred[:, 0]
            return best_func_sym(xp)
        return predict_symbolic
    
    # Try polynomial regression with increasing degrees
    best_poly_r2 = -np.inf
    best_poly_deg = 2
    best_poly_w = None
    
    for deg in range(2, 12):
        if d == 1:
            A = np.vander(x_sorted, N=deg+1, increasing=True)
        else:
            # For multi-dim, use first dimension
            A = np.vander(x_sorted, N=deg+1, increasing=True)
        
        try:
            w, *_ = np.linalg.lstsq(A, y_unique, rcond=None)
            y_hat = A @ w
            sse = np.sum((y_unique - y_hat)**2)
            sst = np.sum((y_unique - np.mean(y_unique))**2)
            r2 = 1 - sse / sst if sst > 1e-12 else 1.0
            
            if r2 > best_poly_r2:
                best_poly_r2 = r2
                best_poly_deg = deg
                best_poly_w = w
                
            if r2 > 0.9999:
                break
        except:
            continue
        if time_left() < 5.0:
            break
    
    # If polynomial fits well, use it
    if best_poly_r2 > 0.999 and best_poly_w is not None:
        def predict_poly(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            xp = X_pred[:, 0]
            A = np.vander(xp, N=best_poly_deg+1, increasing=True)
            return A @ best_poly_w
        return predict_poly
    
    # Fallback: Use PCHIP even if not perfect, or use a simple linear fit
    try:
        pchip = PchipInterpolator(unique_x, y_unique)
        
        def predict_fallback(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            xp = X_pred[:, 0]
            # Clip to training range to avoid extrapolation issues
            xp_clipped = np.clip(xp, unique_x[0], unique_x[-1])
            return pchip(xp_clipped)
        
        return predict_fallback
    except:
        # Last resort: linear fit on first dimension
        if d >= 1:
            x_fit = x_sorted
            A = np.column_stack([x_fit, np.ones_like(x_fit)])
            w, *_ = np.linalg.lstsq(A, y_unique, rcond=None)
            
            def predict_linear(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                xp = X_pred[:, 0]
                A_pred = np.column_stack([xp, np.ones_like(xp)])
                return A_pred @ w
            
            return predict_linear
        else:
            return lambda X: np.full(len(X), y_mean)

# EVOLVE-BLOCK-END
