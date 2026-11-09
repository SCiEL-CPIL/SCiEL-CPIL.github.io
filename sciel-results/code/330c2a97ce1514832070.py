import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed, well-conditioned linear basis using piecewise linear "hat" functions.
    
    This approach uses a fixed linear model y = sum p_i * H_i(x), where H_i are triangular
    hat functions centered at evenly spaced knots in [-2, 2]. This model is strictly linear
    in the parameters, making the optimization problem convex (linear least squares). 
    BFGS will converge rapidly and robustly to the global optimum for this basis.
    
    Unlike Chebyshev polynomials, which can exhibit oscillations (Runge's phenomenon) or 
    struggle with sharp features, hat functions provide a local, non-oscillatory approximation.
    With 10 parameters, we can represent piecewise linear functions with 10 "humps", allowing
    for flexible fitting of various shapes.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    n = 10
    # Define knots evenly spaced in [-2, 2]
    # We want 10 hat functions. Let's place centers at linspace(-2, 2, 10).
    # The width of each hat is determined by the spacing.
    centers = np.linspace(-2.0, 2.0, n)
    # Spacing between centers
    if n > 1:
        width = centers[1] - centers[0]
    else:
        width = 4.0
        
    # Compute hat functions
    # H_i(x) = max(0, 1 - |x - c_i| / width)
    # This creates a triangle with base [c_i - width, c_i + width] and peak 1 at c_i.
    # Note: For the first and last centers, the hats will be truncated by the domain boundaries 
    # effectively, but since we only evaluate on x in [-2, 2], and the first/last centers are at 
    # the boundaries, they will be half-hats (ramps). This is fine and standard for boundary nodes.
    
    # Vectorized computation
    # x shape: (N,)
    # centers shape: (10,)
    # We need (N, 10)
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    
    # Distance from each x to each center
    dist = np.abs(x_col - c_row)
    
    # Hat values
    # 1 - dist/width, clipped to 0
    H = np.maximum(0.0, 1.0 - dist / width)
    
    # Linear combination
    out = H @ p
    
    # Ensure finite
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        
    return out
