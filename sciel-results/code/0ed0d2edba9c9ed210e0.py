import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    # Hard limit is 150s, but we want to be fast. 
    # Ridge regression is O(N * D^2) or similar, very fast for N~10^4, D=6.
    deadline = start_time + 10.0
    
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter out any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n_samples = len(x)
    if n_samples == 0:
        return np.zeros(6)
    
    if n_samples < 6:
        # Not enough data for degree 5, fallback to lower degree or zeros
        # Pad with zeros to return 6 coeffs
        try:
            deg = min(5, n_samples - 1)
            if deg >= 0:
                coeff = np.polynomial.polynomial.polyfit(x, y, deg)
                out = np.zeros(6)
                out[:len(coeff)] = coeff
                return out
        except Exception:
            pass
        return np.zeros(6)
    
    # Construct Vandermonde matrix for degree 5
    # V[i, j] = x[i] ** j
    # Using vander with increasing=True to get ascending powers
    V = np.vander(x, N=6, increasing=True)
    
    # We want to solve (V.T @ V + lambda * I) c = V.T @ y
    # We can search for a small lambda to regularize the fit.
    # The noise in Lorenz-96 subgrid tendencies can be significant.
    # A small amount of regularization prevents the polynomial from fitting noise spikes.
    
    # Heuristic for lambda:
    # If we use lambda=0, it's standard OLS.
    # Let's try a few small lambdas and pick the one with lowest training MSE?
    # No, training MSE will always decrease with higher flexibility (lower lambda).
    # We need a proxy for generalization. 
    # Often, a small fixed lambda like 1e-3 * trace(V.T V) / N works well.
    # Or simply 1e-4 to 1e-2.
    
    # Let's try a small grid search for lambda to minimize a "regularized" training error
    # or just pick a sensible default. 
    # Given the "unresolved" nature, the signal is likely smooth.
    # Let's use 1e-4 as a baseline.
    
    # To be safe and potentially better, let's try 3 values: 0.0, 1e-4, 1e-3.
    # We can't validate on training data for selection (overfitting).
    # However, Ridge regression is convex. 
    # Let's use a single, small regularization parameter.
    # The scale of x matters. Let's normalize x for the fit, then transform back?
    # No, the contract requires coefficients in powers of the *original* x.
    # If x is large, high powers are huge. 
    # Let's check the range of x.
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    if x_range == 0:
        return np.zeros(6)
    
    # Normalizing x helps numerical stability of the Vandermonde matrix.
    # Let x_norm = (x - x_mean) / x_std
    # Fit c_norm for x_norm.
    # Then transform c_norm to c for x.
    # This is complex. 
    # Alternative: Use SVD to solve V c = y. It's more stable than normal equations.
    
    # Let's use SVD.
    try:
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        # Pseudo-inverse
        # c = Vt.T @ np.diag(1/s) @ U.T @ y
        # To regularize, we can truncate small singular values or use Tikhonov.
        
        # Tikhonov regularization with SVD:
        # c = Vt.T @ diag(s / (s^2 + lambda)) @ U.T @ y
        
        # Let's pick lambda based on the singular values.
        # A common choice is lambda = s_min or a fraction of s_max.
        # Let's try lambda = 1e-6 * max(s)
        
        s_max = s[0] if len(s) > 0 else 1.0
        lam = 1e-6 * s_max
        
        # Compute coefficients
        # y_proj = U.T @ y
        y_proj = U.T @ y
        # c = Vt.T @ (s / (s**2 + lam)) * y_proj
        coeffs_scaled = s / (s**2 + lam)
        c = Vt.T @ (coeffs_scaled * y_proj)
        
        if not np.all(np.isfinite(c)):
            # Fallback to OLS
            c = np.polynomial.polynomial.polyfit(x, y, 5)
            
        return c
        
    except Exception:
        # Fallback to standard polyfit if SVD fails
        try:
            return np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            return np.zeros(6)
