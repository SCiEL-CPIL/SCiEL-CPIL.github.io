"""Symbolic regression with Piecewise Cubic Spline basis."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed knots for the spline: 5 equally spaced knots in [-2, 2]
# The 10 params are split: first 5 are y-values at the 5 knots,
# remaining 5 are unused (or we can use all 10 for 10 knots)
# Let's use 10 knots for better flexibility with 10 params

_KNOTS = np.linspace(-2.0, 2.0, 10)

def _cubic_spline_eval(x: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    """
    Evaluate a natural cubic spline at points x given knot values y_knots.
    
    We use the standard cubic spline interpolation with natural boundary conditions
    (zero second derivative at endpoints).
    
    Parameters:
    -----------
    x : np.ndarray, shape (N,)
        Points at which to evaluate the spline.
    y_knots : np.ndarray, shape (K,)
        Function values at the K knots.
    
    Returns:
    --------
    np.ndarray, shape (N,)
        Spline values at x.
    """
    K = len(y_knots)
    if K < 2:
        return np.full_like(x, y_knots[0] if K == 1 else 0.0)
    
    # Knot positions
    t = _KNOTS
    h = np.diff(t)  # shape (K-1,)
    
    # Solve for second derivatives at interior knots using natural boundary conditions
    # The system is tridiagonal. We'll use the standard algorithm.
    
    if K == 2:
        # Linear interpolation for 2 knots
        slope = (y_knots[1] - y_knots[0]) / (t[1] - t[0])
        return y_knots[0] + slope * (x - t[0])
    
    # Natural cubic spline: M[0] = 0, M[K-1] = 0
    # For interior knots i = 1, ..., K-2:
    # h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    
    # Build tridiagonal system
    n = K - 2  # number of interior knots
    if n == 0:
        # Only 2 knots, handled above
        pass
    
    # Use Thomas algorithm for tridiagonal solve
    # a[i], b[i], c[i] for lower, diagonal, upper
    b = np.zeros(n)
    c = np.zeros(n)
    a = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        idx = i + 1  # interior knot index
        if i > 0:
            a[i] = h[idx - 1]
        if i < n - 1:
            c[i] = h[idx]
        b[i] = 2.0 * (h[idx - 1] + h[idx])
        d[i] = 6.0 * ((y_knots[idx + 1] - y_knots[idx]) / h[idx] - (y_knots[idx] - y_knots[idx - 1]) / h[idx - 1])
    
    # Thomas algorithm
    if n == 0:
        M = np.zeros(K)
    else:
        # Forward elimination
        cp = np.zeros(n)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0] if b[0] != 0 else 0.0
        dp[0] = d[0] / b[0] if b[0] != 0 else 0.0
        
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1] if i > 0 else b[i]
            if i < n - 1:
                cp[i] = c[i] / denom if denom != 0 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom if denom != 0 else 0.0
        
        # Back substitution
        M_interior = np.zeros(n)
        M_interior[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            M_interior[i] = dp[i] - cp[i] * M_interior[i + 1]
        
        M = np.zeros(K)
        M[1:-1] = M_interior
        # M[0] = 0, M[K-1] = 0 (natural boundary)
    
    # Now evaluate the spline at x
    # For each x, find which interval it falls in, then evaluate the cubic polynomial
    
    # Find interval indices
    # x is in [t[i], t[i+1]] for some i
    # Use searchsorted
    idx = np.searchsorted(t, x, side='right') - 1
    idx = np.clip(idx, 0, K - 2)
    
    # For each point, we have interval i, so:
    # S(x) = M[i]*((t[i+1]-x)^3)/(6*h[i]) + M[i+1]*((x-t[i])^3)/(6*h[i]) + (y[i]-M[i]*h[i]^2/6)*((t[i+1]-x)/h[i]) + (y[i+1]-M[i+1]*h[i]^2/6)*((x-t[i])/h[i])
    
    ti = t[idx]
    ti1 = t[idx + 1]
    hi = h[idx]
    yi = y_knots[idx]
    yi1 = y_knots[idx + 1]
    Mi = M[idx]
    Mi1 = M[idx + 1]
    
    dx1 = ti1 - x
    dx2 = x - ti
    
    result = (Mi * dx1**3 / (6.0 * hi) + 
              Mi1 * dx2**3 / (6.0 * hi) + 
              (yi - Mi * hi**2 / 6.0) * dx1 / hi + 
              (yi1 - Mi1 * hi**2 / 6.0) * dx2 / hi)
    
    return result


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a natural cubic spline basis.
    
    The 10 params are the y-values at 10 fixed, equally spaced knots in [-2, 2].
    A natural cubic spline (zero second derivative at boundaries) interpolates these points.
    
    This local, flexible representation can model sharp features and non-polynomial
    behavior better than a global polynomial or RBF basis.
    """
    x = np.asarray(x, dtype=float)
    
    # params shape: (10,), knots shape: (10,)
    y_knots = params[:10]
    
    return _cubic_spline_eval(x, y_knots)
# EVOLVE-BLOCK-END
