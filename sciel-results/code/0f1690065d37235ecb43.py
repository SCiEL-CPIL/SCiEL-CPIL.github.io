"""Piecewise cubic spline basis with natural boundary conditions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a piecewise cubic spline representation.
    
    The 10 parameters define:
    - params[0:5]: y-values at 5 equally spaced knots in [-2, 2]
    - params[5:10]: shape control parameters that modify the spline's second derivatives
    
    Natural cubic spline (zero second derivative at endpoints) is used for interpolation.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Define 5 equally spaced knots in [-2, 2]
    knots = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Extract y-values at knots and shape parameters
    y_knots = params[0:5].copy()
    shape_params = params[5:10]
    
    # Compute natural cubic spline coefficients
    # Using the standard algorithm for natural cubic splines
    
    # Segment lengths
    h = np.diff(knots)  # [1, 1, 1, 1]
    
    # Set up the tridiagonal system for second derivatives (moments)
    # Natural boundary conditions: M[0] = 0, M[4] = 0
    # For interior points i=1,2,3:
    # h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*(f[i+1]-f[i])/h[i] - 6*(f[i]-f[i-1])/h[i-1]
    
    M = np.zeros(5)  # Second derivatives at knots
    
    # Compute right-hand side for interior points
    rhs = np.zeros(3)
    for i in range(1, 4):
        rhs[i-1] = 6 * ((y_knots[i+1] - y_knots[i]) / h[i] - (y_knots[i] - y_knots[i-1]) / h[i-1])
    
    # Solve the tridiagonal system
    # The matrix is:
    # [2*(h0+h1), h1, 0]
    # [h1, 2*(h1+h2), h2]
    # [0, h2, 2*(h2+h3)]
    # Since all h[i] = 1, this simplifies to:
    # [4, 1, 0]
    # [1, 4, 1]
    # [0, 1, 4]
    
    # Solve using direct inversion for this small system
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ])
    
    # Add shape parameters to modify the spline behavior
    # shape_params can adjust the moments to allow for more flexible shapes
    rhs_modified = rhs + shape_params[0:3]
    
    M[1:4] = np.linalg.solve(A, rhs_modified)
    
    # Compute spline coefficients for each segment
    # For segment i (from knots[i] to knots[i+1]):
    # S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    
    a = np.zeros(4)
    b = np.zeros(4)
    c = np.zeros(4)
    d = np.zeros(4)
    
    for i in range(4):
        a[i] = y_knots[i]
        c[i] = M[i] / 2.0
        d[i] = (M[i+1] - M[i]) / (6.0 * h[i])
        b[i] = (y_knots[i+1] - y_knots[i]) / h[i] - h[i] * (2.0 * M[i] + M[i+1]) / 6.0
    
    # Add additional shape parameters to adjust the spline
    # shape_params[3:5] can be used to add cubic corrections
    cubic_correction = shape_params[3:5]
    
    # Evaluate the spline at each x
    result = np.zeros(n)
    
    for j in range(n):
        xj = x[j]
        
        # Find which segment xj falls in
        if xj <= knots[0]:
            # Extrapolate using the first segment
            t = xj - knots[0]
            result[j] = a[0] + b[0]*t + c[0]*t**2 + d[0]*t**3 + cubic_correction[0]*t**3
        elif xj >= knots[4]:
            # Extrapolate using the last segment
            t = xj - knots[3]
            result[j] = a[3] + b[3]*t + c[3]*t**2 + d[3]*t**3 + cubic_correction[1]*t**3
        else:
            # Find the segment
            for i in range(4):
                if xj <= knots[i+1]:
                    t = xj - knots[i]
                    result[j] = a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3
                    break
    
    return result
# EVOLVE-BLOCK-END
