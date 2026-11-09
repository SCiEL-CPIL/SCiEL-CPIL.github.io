"""
Strategy: Fixed-Basis Polynomial Regression (Degree 9).

Rationale:
1. The task is symbolic regression with 1 input.
2. We have 10 parameters available.
3. The previous linear model (degree 1) is too restrictive for non-linear targets.
4. Non-linear basis functions (like RBFs) require optimizing centers/widths, which
   leads to non-convex optimization landscapes where BFGS often gets stuck in local minima
   or converges slowly, especially starting from all ones.
5. A polynomial basis $1, x, x^2, ..., x^9$ makes the model linear in the parameters.
   The objective function (MSE) becomes a convex quadratic function of the parameters.
   BFGS is highly effective for convex problems and will converge to the global minimum
   (the best least-squares fit) quickly and reliably.
6. A 9th-degree polynomial can approximate a wide variety of smooth functions on [-3, 3].
   While it may oscillate (Runge's phenomenon) at the edges, for a general unknown function
   it provides significantly more expressive power than a linear model, potentially capturing
   curvature, saturation, or periodic-like behavior within the range.
7. To ensure numerical stability, we use a simple power basis. The inputs are in [-3, 3],
   so $x^9$ is at most $19683$, which is well within float64 precision.
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a 9th-degree polynomial.
    
    params[0] corresponds to the coefficient for x^0 (1)
    params[1] corresponds to the coefficient for x^1 (x)
    ...
    params[9] corresponds to the coefficient for x^9
    
    This structure ensures the model is linear in params, making the optimization landscape convex.
    """
    # x is an array of shape (N,). We need to compute the polynomial for each point.
    # We can use Horner's method for efficiency and numerical stability, 
    # or simply sum the terms. Given N=200 and degree 9, direct summation is fine.
    
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    # Initialize result with the constant term (params[0] * 1)
    # We build the polynomial: p0 + p1*x + p2*x^2 + ... + p9*x^9
    
    # Using a loop for clarity and to avoid intermediate large arrays if x is large, 
    # though vectorized is also fast enough for N=200.
    result = np.zeros_like(x, dtype=np.float64)
    
    # Power of x
    x_power = np.ones_like(x, dtype=np.float64)
    
    for i in range(10):
        result += params[i] * x_power
        x_power *= x
        
    return result
# EVOLVE-BLOCK-END
