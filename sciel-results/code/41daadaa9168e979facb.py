import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Piecewise Cubic Spline basis.
    
    Strategy:
    - The 10 parameters are split into two groups:
      1. 5 parameters (params[0:5]) represent the y-values of the function at 5 fixed,
         equally spaced knots across [-2, 2]. These define a cubic spline with natural
         boundary conditions (zero second derivative).
      2. 5 parameters (params[5:10]) represent coefficients for a global linear trend
         component (1, x, x^2, x^3, x^4) to capture any residual global structure not
         well-represented by the local spline knots, or to stabilize the fit if the
         spline knots are poorly placed.
    - This hybrid approach combines the local flexibility of splines (good for sharp
      features or non-smooth behavior) with the global stability of low-degree polynomials.
    - The spline evaluation is done via pre-computed basis functions (B-splines or
      direct cubic Hermite interpolation) for efficiency.
    """
    start_time = time.time()
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
        
    # Deadline guard: if we are already too close to the limit, return zeros
    # (though this function is very fast, it's a safety net)
    if time.time() - start_time > 110.0:
        return np.zeros(n)

    # Split parameters
    # params[0:5] -> Spline knot values y0..y4
    # params[5:10] -> Global polynomial coefficients c0..c4 for 1, x, x^2, x^3, x^4
    
    y_knots = params[0:5]
    poly_coeffs = params[5:10]
    
    # Define fixed knots in [-2, 2]
    # 5 knots: -2.0, -1.0, 0.0, 1.0, 2.0
    knots = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Calculate spline values
    # We use a simple cubic spline evaluation.
    # Since the knots are fixed and known, we can pre-compute the spline coefficients
    # (second derivatives) based on y_knots.
    
    # For a natural cubic spline, the second derivatives M_i at knots i=0..n satisfy:
    # M_0 = 0, M_n = 0
    # h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1} = 6 * ( (y_{i+1} - y_i)/h_i - (y_i - y_{i-1})/h_{i-1} )
    
    # Here h_i = knots[i+1] - knots[i] = 1.0 for all i
    # So the system is:
    # 2(1+1) M_0 + 1 M_1 = ... but M_0=0, so 2 M_1 = 6 * ((y_1 - y_0)/1 - (y_0 - y_{-1})/1) -- wait, boundary conditions are different.
    
    # Standard Natural Cubic Spline with n+1 points (here 5 points, indices 0..4):
    # M_0 = 0
    # M_4 = 0
    # For i=1: h_0 M_0 + 2(h_0+h_1) M_1 + h_1 M_2 = 6 * ( (y_2 - y_1)/h_1 - (y_1 - y_0)/h_0 )
    #   => 0 + 2(2) M_1 + 1 M_2 = 6 * ( y_2 - y_1 - (y_1 - y_0) ) = 6 * ( y_2 - 2y_1 + y_0 )
    # For i=2: h_1 M_1 + 2(h_1+h_2) M_2 + h_2 M_3 = 6 * ( (y_3 - y_2)/h_2 - (y_2 - y_1)/h_1 )
    #   => 1 M_1 + 4 M_2 + 1 M_3 = 6 * ( y_3 - 2y_2 + y_1 )
    # For i=3: h_2 M_2 + 2(h_2+h_3) M_3 + h_3 M_4 = 6 * ( (y_4 - y_3)/h_3 - (y_3 - y_2)/h_2 )
    #   => 1 M_2 + 4 M_3 + 0 = 6 * ( y_4 - 2y_3 + y_2 )
    
    # System:
    # 4 M_1 + M_2 = 6 * (y_2 - 2y_1 + y_0)
    # M_1 + 4 M_2 + M_3 = 6 * (y_3 - 2y_2 + y_1)
    # M_2 + 4 M_3 = 6 * (y_4 - 2y_3 + y_2)
    
    # This is a small 3x3 linear system for M_1, M_2, M_3.
    # We can solve it analytically or use np.linalg.solve. Since it's tiny, solve is fast enough.
    
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ])
    
    rhs = 6.0 * np.array([
        y_knots[2] - 2.0 * y_knots[1] + y_knots[0],
        y_knots[3] - 2.0 * y_knots[2] + y_knots[1],
        y_knots[4] - 2.0 * y_knots[3] + y_knots[2]
    ])
    
    try:
        M_interior = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # Fallback if singular (should not happen with this A)
        M_interior = np.zeros(3)
        
    M = np.zeros(5)
    M[1] = M_interior[0]
    M[2] = M_interior[1]
    M[3] = M_interior[2]
    # M[0] = 0, M[4] = 0
    
    # Evaluate spline for each x
    # Spline formula on interval [x_i, x_{i+1}]:
    # S(x) = M_i * (x_{i+1} - x)^3 / (6 h) + M_{i+1} * (x - x_i)^3 / (6 h) + (y_i / h - M_i * h / 6) * (x_{i+1} - x) + (y_{i+1} / h - M_{i+1} * h / 6) * (x - x_i)
    # Here h = 1.0 always.
    
    result_spline = np.zeros(n)
    
    # Vectorized evaluation
    # Determine interval index for each x
    # x in [-2, -1] -> i=0
    # x in [-1, 0] -> i=1
    # x in [0, 1] -> i=2
    # x in [1, 2] -> i=3
    
    # Clip x to [knots[0], knots[-1]] for safety, though inputs are in [-2,2]
    x_clipped = np.clip(x, -2.0, 2.0)
    
    # Interval indices
    idx = np.floor((x_clipped + 2.0)).astype(int)
    # If x=2.0, idx=4, but we only have intervals 0..3.
    # For x=2.0, we should use interval 3.
    idx = np.clip(idx, 0, 3)
    
    # Local coordinates within interval
    # x_i = -2.0 + idx
    x_i = -2.0 + idx
    x_ip1 = x_i + 1.0
    h = 1.0
    
    # Terms
    # Term 1: M_i * (x_{i+1} - x)^3 / 6
    # Term 2: M_{i+1} * (x - x_i)^3 / 6
    # Term 3: (y_i / h - M_i * h / 6) * (x_{i+1} - x)
    # Term 4: (y_{i+1} / h - M_{i+1} * h / 6) * (x - x_i)
    
    # We need to gather M_i, M_{i+1}, y_i, y_{i+1} for each point
    # M is length 5, idx is 0..3
    M_i = M[idx]
    M_ip1 = M[idx + 1]
    y_i = y_knots[idx]
    y_ip1 = y_knots[idx + 1]
    
    dx1 = x_ip1 - x_clipped
    dx2 = x_clipped - x_i
    
    term1 = M_i * (dx1**3) / 6.0
    term2 = M_ip1 * (dx2**3) / 6.0
    term3 = (y_i - M_i / 6.0) * dx1
    term4 = (y_ip1 - M_ip1 / 6.0) * dx2
    
    result_spline = term1 + term2 + term3 + term4
    
    # Add global polynomial component
    # params[5:10] are coefficients for 1, x, x^2, x^3, x^4
    # This allows the model to capture global trends if the spline knots are not sufficient
    # or to shift the entire curve.
    
    poly_term = (poly_coeffs[0] * np.ones(n) + 
                 poly_coeffs[1] * x + 
                 poly_coeffs[2] * (x**2) + 
                 poly_coeffs[3] * (x**3) + 
                 poly_coeffs[4] * (x**4))
                 
    result = result_spline + poly_term
    
    if not np.all(np.isfinite(result)):
        return np.zeros(n)
        
    return result
