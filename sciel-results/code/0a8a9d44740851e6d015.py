import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Complete cubic polynomial model for a nonlinear oscillator.
    
    This structural change moves away from assumed physical terms (tanh, sin)
    to a general-purpose polynomial basis. This allows the optimizer to discover
    the functional form directly from the data without being constrained by
    potentially incorrect physical priors (like specific damping structures).
    
    The inputs are normalized to improve the conditioning of the polynomial terms,
    which is crucial for BFGS convergence when starting from arbitrary initial parameters.
    
    Basis terms (10 total):
    1. 1
    2. x
    3. v
    4. x^2
    5. x*v
    6. v^2
    7. x^3
    8. x^2*v
    9. x*v^2
    10. v^3
    """
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> mean ~ -0.02265, scale ~ 0.55015 (half-range)
    # v: [-0.1754, 0.2821] -> mean ~ 0.05335, scale ~ 0.22875 (half-range)
    
    x_mean = -0.02265
    x_scale = 0.55015
    v_mean = 0.05335
    v_scale = 0.22875
    
    # Normalize inputs to approximately [-1, 1]
    x_n = (x - x_mean) / x_scale
    v_n = (v - v_mean) / v_scale
    
    p = params
    
    # Construct basis terms
    # Linear
    t1 = np.ones_like(x_n)
    t2 = x_n
    t3 = v_n
    
    # Quadratic
    t4 = x_n * x_n
    t5 = x_n * v_n
    t6 = v_n * v_n
    
    # Cubic
    t7 = x_n * x_n * x_n
    t8 = x_n * x_n * v_n
    t9 = x_n * v_n * v_n
    t10 = v_n * v_n * v_n
    
    # Combine
    # a = sum(p_i * t_i)
    a = (p[0] * t1 + 
         p[1] * t2 + 
         p[2] * t3 + 
         p[3] * t4 + 
         p[4] * t5 + 
         p[5] * t6 + 
         p[6] * t7 + 
         p[7] * t8 + 
         p[8] * t9 + 
         p[9] * t10)
         
    # Scale back to original acceleration units
    # The target y is in the range [-0.1047, 0.1106].
    # The normalized polynomial output will be in the range of the coefficients.
    # We don't need to scale back explicitly if the coefficients learn the right magnitude,
    # but ensuring the output is in the correct "units" helps BFGS if the loss is MSE.
    # Since we are fitting parameters, the scale is absorbed into the parameters.
    # However, if the normalized basis values are O(1), the coefficients will be O(0.1).
    # This is numerically stable.
    
    return a
