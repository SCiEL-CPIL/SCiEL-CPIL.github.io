"""
Symbolic regression candidate using a Composite Non-Linear Basis.

Strategy:
The previous approaches relied on linear combinations of fixed basis functions 
(Chebyshev, RBFs). While stable, they are limited in the functional forms they 
can represent efficiently. 

This strategy uses a hybrid model:
1. A cubic polynomial (4 params) to capture global trends.
2. A hyperbolic tangent term (2 params: amplitude, center) to capture bounded 
   S-shaped transitions or asymptotic behaviors.
3. A Gaussian RBF term (3 params: amplitude, center, width) to capture local 
   peaks or dips.
4. A constant term (1 param) for bias.

Total parameters: 10.

The key structural difference is that the basis functions themselves depend on 
the parameters (e.g., the RBF width and center are parameters). This makes the 
optimization problem non-linear. BFGS can navigate this landscape, and the 
initialization (all ones) maps to a reasonable "flat" state:
- Poly: x^3 + x^2 + x + 1
- Tanh: 1 * tanh(x - 1)
- RBF: 1 * exp(-(x - 1)^2 / (2 * 1^2))

This allows the model to adapt its shape to the target function, potentially 
capturing features that fixed-basis models miss.
"""
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate.
    
    Args:
        x: Input array.
        params: Array of 10 parameters.
        
    Returns:
        Predictions array.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n_points = len(x)
    
    if n_points == 0:
        return np.array([])

    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]

    # Extract parameters
    # 0-3: Polynomial coefficients (c0, c1, c2, c3)
    c0, c1, c2, c3 = p[0], p[1], p[2], p[3]
    
    # 4-5: Tanh parameters (amplitude, center)
    a_tanh, c_tanh = p[4], p[5]
    
    # 6-8: RBF parameters (amplitude, center, width)
    a_rbf, c_rbf, w_rbf = p[6], p[7], p[8]
    
    # 9: Bias
    bias = p[9]

    # 1. Polynomial part
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3

    # 2. Tanh part
    # Ensure numerical stability for large arguments
    # tanh is bounded, so no overflow, but we can clip if needed for extreme inputs
    # x is in [-3, 3], so x - c_tanh is small.
    tanh_term = a_tanh * np.tanh(x - c_tanh)

    # 3. RBF part
    # Width must be positive to avoid division by zero or negative variance
    # We use abs(w_rbf) + epsilon to ensure it's positive and non-zero
    # If w_rbf is close to 0, the RBF becomes a spike. 
    # To prevent numerical instability (overflow in exp), we clip the exponent.
    # exp(-large) -> 0, which is fine.
    # But if w_rbf is very small, (x-c)^2 / w^2 can be huge.
    # Let's enforce a minimum width to keep it smooth and stable for BFGS.
    min_width = 0.1
    w = np.abs(w_rbf) + min_width
    
    diff = x - c_rbf
    # Clip the argument to exp to avoid underflow warnings (though exp(-1000) is just 0)
    # and to prevent potential issues if w was tiny (handled by min_width).
    arg = -(diff ** 2) / (2.0 * w ** 2)
    # Clip arg to [-500, 0] to ensure exp is in [0, 1] and finite
    arg = np.clip(arg, -500.0, 0.0)
    rbf_term = a_rbf * np.exp(arg)

    # Combine
    pred = poly + tanh_term + rbf_term + bias

    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
