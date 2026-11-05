"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Uses robust polynomial regression with multiple strategies and selects
    the best performing polynomial based on training fit quality.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Remove any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Strategy 1: Direct least-squares fit of degree 5
    def fit_degree5_direct(x, y):
        try:
            # Use np.polynomial.polynomial.polyfit which uses least squares
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return None
    
    # Strategy 2: Ridge regression with various alpha values
    def fit_ridge(x, y, alpha):
        try:
            # Build Vandermonde matrix in ascending powers
            V = np.vander(x, N=6, increasing=True)
            # Ridge: (V^T V + alpha*I)^{-1} V^T y
            I = np.eye(6)
            A = V.T @ V + alpha * I
            b = V.T @ y
            coeff = np.linalg.solve(A, b)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return None
    
    # Strategy 3: Polynomial fit on scaled x
    def fit_scaled(x, y, degree=5):
        try:
            x_min = np.min(x)
            x_max = np.max(x)
            x_range = x_max - x_min
            if x_range < 1e-12:
                return None
            # Scale to [-1, 1]
            x_scaled = 2.0 * (x - x_min) / x_range - 1.0
            coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, degree)
            if not np.all(np.isfinite(coeff_scaled)):
                return None
            # Convert back: p(x) where x_scaled = 2*(x-xmin)/xrange - 1
            # Need to expand: coeff_scaled[i] * (2*(x-xmin)/xrange - 1)^i
            # This is complex, so use a different approach:
            # Just return the scaled fit and evaluate it properly
            # Actually, let's just do the conversion via polynomial arithmetic
            # For now, let's use a simpler approach
            return None
        except Exception:
            pass
        return None
    
    # Strategy 4: Weighted least squares with downweighting of outliers
    def fit_weighted(x, y):
        try:
            # First do an initial fit to estimate residuals
            coeff_init = np.polynomial.polynomial.polyfit(x, y, 3)
            if not np.all(np.isfinite(coeff_init)):
                return None
            y_pred = np.polynomial.polynomial.polyval(x, coeff_init)
            residuals = y - y_pred
            # Use Huber-like weights: downweight large residuals
            sigma = np.median(np.abs(residuals)) + 1e-12
            w = 1.0 / (1.0 + (residuals / (2.0 * sigma))**2)
            # Weighted least squares
            V = np.vander(x, N=6, increasing=True)
            W = np.diag(w)
            A = V.T @ W @ V + 1e-10 * np.eye(6)
            b = V.T @ W @ y
            coeff = np.linalg.solve(A, b)
            if np.all(np.isfinite(coeff)):
                return coeff
        except Exception:
            pass
        return None
    
    # Evaluate a coefficient set by computing normalized MSE on training data
    def eval_coeff(coeff, x, y):
        try:
            y_pred = np.polynomial.polynomial.polyval(x, coeff)
            mse = np.mean((y - y_pred)**2)
            var = np.var(y)
            if var < 1e-15:
                return mse
            return mse / var
        except Exception:
            return np.inf
    
    # Collect candidate coefficient sets
    candidates = []
    
    # Direct degree 5
    c = fit_degree5_direct(x, y)
    if c is not None:
        candidates.append(('direct5', c))
    
    # Ridge with various alphas
    for alpha in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
        c = fit_ridge(x, y, alpha)
        if c is not None:
            candidates.append((f'ridge_{alpha}', c))
    
    # Weighted fit
    c = fit_weighted(x, y)
    if c is not None:
        candidates.append(('weighted', c))
    
    # Also try lower degrees embedded in degree-5 space (higher coeffs = 0)
    for deg in [1, 2, 3, 4]:
        try:
            coeff_low = np.polynomial.polynomial.polyfit(x, y, deg)
            if np.all(np.isfinite(coeff_low)):
                c = np.zeros(6)
                c[:deg+1] = coeff_low
                candidates.append((f'deg{deg}', c))
        except Exception:
            pass
    
    if not candidates:
        # Fallback: simple linear
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            c = np.zeros(6)
            c[:2] = coeff
            return c
        except Exception:
            return np.zeros(6)
    
    # Score all candidates
    best_score = np.inf
    best_coeff = None
    for name, c in candidates:
        score = eval_coeff(c, x, y)
        if score < best_score:
            best_score = score
            best_coeff = c.copy()
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        if best_coeff is not None:
            return best_coeff
        return np.zeros(6)
    
    if best_coeff is not None:
        return best_coeff
    return np.zeros(6)
# EVOLVE-BLOCK-END
