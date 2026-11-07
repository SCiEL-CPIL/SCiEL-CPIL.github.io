# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis:
    - 5 Chebyshev polynomial terms (degrees 0-4) on mapped domain [-1, 1]
    - Sinusoidal terms with learnable frequency: sin(w*x), cos(w*x)
    - Exponential term with learnable rate: exp(b*x)
    
    params layout (length 10):
      p[0]: coeff for T0(t)
      p[1]: coeff for T1(t)
      p[2]: coeff for T2(t)
      p[3]: coeff for T3(t)
      p[4]: coeff for T4(t)
      p[5]: coeff for sin(w*x)
      p[6]: coeff for cos(w*x)
      p[7]: frequency w
      p[8]: coeff for exp(b*x)
      p[9]: rate b
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev Polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t2 - 1.0
    b3 = 4.0 * t3 - 3.0 * t
    b4 = 8.0 * t4 - 8.0 * t2 + 1.0
    
    # Sinusoidal terms
    # Clamp frequency to avoid excessive oscillation/aliasing during optimization
    w = np.clip(params[7], -10.0, 10.0)
    b5 = np.sin(w * x)
    b6 = np.cos(w * x)
    
    # Exponential term
    # Clamp rate to prevent overflow/underflow and maintain gradient stability
    b_rate = np.clip(params[9], -5.0, 5.0)
    exp_arg = b_rate * x
    # Clamp argument to safe range for exp
    exp_arg = np.clip(exp_arg, -50.0, 50.0)
    b7 = np.exp(exp_arg)
    
    # We have 8 basis functions so far. We need 10 params.
    # The prompt asks for "a single learnable-frequency sinusoidal term" and "a learnable exponential term".
    # My current basis has:
    # 5 Chebyshev
    # 2 Sin/Cos (shared freq) -> counts as 1 "sinusoidal term" structure? Or is it 2 terms?
    # 1 Exp
    # Total basis functions: 8.
    # Total params: 5 (cheb) + 2 (sin/cos coeffs) + 1 (freq) + 1 (exp coeff) + 1 (exp rate) = 10.
    # This fits perfectly.
    
    # Stack into matrix
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7])
    
    # We need to assign params to these 8 columns.
    # p[0] -> b0
    # p[1] -> b1
    # p[2] -> b2
    # p[3] -> b3
    # p[4] -> b4
    # p[5] -> b5
    # p[6] -> b6
    # p[8] -> b7
    # p[7] is used in b5, b6 calculation
    # p[9] is used in b7 calculation
    
    # Construct the weighted sum directly to avoid creating a dummy column for w and b_rate
    # which are non-linear params embedded in the basis.
    
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
              p5 * b5 + p6 * b6 + p8 * b7)
    
    # Wait, I need to extract params first.
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Recalculate with extracted params to be clean
    w = np.clip(p7, -10.0, 10.0)
    b_rate = np.clip(p9, -5.0, 5.0)
    
    b5 = np.sin(w * x)
    b6 = np.cos(w * x)
    exp_arg = np.clip(b_rate * x, -50.0, 50.0)
    b7 = np.exp(exp_arg)
    
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
              p5 * b5 + p6 * b6 + p8 * b7)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
