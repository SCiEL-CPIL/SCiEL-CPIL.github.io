"""
Strategy: Linearly Parameterized Polynomial Basis Expansion.

Rationale:
The previous mechanistic model (Monod * Gaussian * Quadratic) created a highly non-convex
loss landscape for the 10 parameters, causing the external BFGS optimizer to converge to
poor local minima (score ~40). This new approach uses a linear combination of basis functions:
y = sum( w_i * phi_i(x) )

The basis functions are fixed and designed to capture:
1. Intercept
2. Linear effects of all 4 inputs
3. Quadratic effects of temperature and pH (to model optimal ranges)
4. Key interactions between substrate and environment
5. Substrate saturation non-linearity

Since the model is linear in the parameters (weights), the loss function is convex
with respect to the parameters. This allows the BFGS optimizer to reliably find the
global minimum, avoiding the pitfalls of the multiplicative structure.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a linear combination of polynomial basis functions.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Params (length 10):
    p[0]: Intercept
    p[1]: Coeff for b_n
    p[2]: Coeff for s_n
    p[3]: Coeff for t_n
    p[4]: Coeff for ph_n
    p[5]: Coeff for t_n^2
    p[6]: Coeff for ph_n^2
    p[7]: Coeff for s_n * t_n
    p[8]: Coeff for s_n * ph_n
    p[9]: Coeff for s_n^2
    """
    # Normalize inputs to [0, 1] based on task description ranges
    # This stabilizes the polynomial terms and helps the optimizer
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0

    p = params

    # Construct the linear combination of basis functions
    # Basis 0: Constant
    term_0 = np.ones_like(b_n)
    
    # Basis 1-4: Linear terms
    term_1 = b_n
    term_2 = s_n
    term_3 = t_n
    term_4 = ph_n
    
    # Basis 5-6: Quadratic terms for environmental factors
    term_5 = t_n ** 2
    term_6 = ph_n ** 2
    
    # Basis 7-8: Interaction terms
    term_7 = s_n * t_n
    term_8 = s_n * ph_n
    
    # Basis 9: Substrate non-linearity
    term_9 = s_n ** 2

    # Combine
    result = (p[0] * term_0 + 
              p[1] * term_1 + 
              p[2] * term_2 + 
              p[3] * term_3 + 
              p[4] * term_4 + 
              p[5] * term_5 + 
              p[6] * term_6 + 
              p[7] * term_7 + 
              p[8] * term_8 + 
              p[9] * term_9)

    return result
# EVOLVE-BLOCK-END
