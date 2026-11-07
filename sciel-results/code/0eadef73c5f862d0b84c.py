import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a hybrid basis to capture various functional forms:
    - Constant
    - Linear
    - Quadratic
    - Sine and Cosine (low frequency)
    - Exponential (damped)
    - Rational (1/(1+x^2) type)
    - Logarithmic (soft)
    - Step-like (tanh)
    - Cubic
    """
    # Ensure x is a 1D array for consistent shape handling if needed, though x is usually 1D
    x = np.asarray(x, dtype=np.float64)
    
    # Handle potential edge cases for log and division
    # x is in [-2, 2]
    
    # 1. Constant
    f0 = np.ones_like(x)
    
    # 2. Linear
    f1 = x
    
    # 3. Quadratic
    f2 = x * x
    
    # 4. Cubic
    f3 = x * x * x
    
    # 5. Sine
    f4 = np.sin(x)
    
    # 6. Cosine
    f5 = np.cos(x)
    
    # 7. Exponential (damped to avoid overflow/underflow issues, though x is small)
    # exp(-0.5 * x^2) is a Gaussian-like feature
    f6 = np.exp(-0.5 * x * x)
    
    # 8. Rational function 1 / (1 + x^2)
    # This captures Lorentzian-like shapes
    f7 = 1.0 / (1.0 + x * x)
    
    # 9. Logarithmic softness: log(1 + x^2)
    # This is smooth and captures growth
    f8 = np.log1p(x * x)
    
    # 10. Hyperbolic tangent: tanh(x)
    # Captures step-like or saturation behavior
    f9 = np.tanh(x)
    
    # Stack into a matrix of shape (N, 10)
    basis = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    return basis @ params
