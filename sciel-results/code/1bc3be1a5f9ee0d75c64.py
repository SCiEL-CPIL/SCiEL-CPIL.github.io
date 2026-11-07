# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (degree 0-4) + learnable sinusoid + learnable exponential.
    
    The model is:
    f(x) = sum_{k=0}^{4} c_k * T_k(t)  +  A * sin(w * x)  +  B * exp(C * x)
    where t = x / 3.0
    
    - Chebyshev part: linear in c_k, well-conditioned, captures smooth polynomial-like trends.
    - Sinusoid: A, w are learnable (w via tanh scaling to keep bounded).
    - Exponential: B, C are learnable (C via tanh scaling to prevent overflow).
    
    Total parameters: 10 (5 Chebyshev coeffs + A + w_raw + B + C_raw).
    
    Note: This model is NON-linear in params due to sin(w*x) and exp(C*x).
    BFGS will be used to optimize. Starting from all ones, tanh(1) ~ 0.76, so
    w ~ 3.8, C ~ 1.5. These are reasonable initial frequencies/rates.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract parameters
    p = params[:10]
    # Ensure finite
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Chebyshev coefficients
    c0, c1, c2, c3, c4 = p[0], p[1], p[2], p[3], p[4]
    
    # Sinusoid parameters
    A = p[5]
    w_raw = p[6]
    w = np.tanh(w_raw) * 5.0  # frequency in [-5, 5]
    
    # Exponential parameters
    B = p[7]
    C_raw = p[8]
    C = np.tanh(C_raw) * 2.0  # rate in [-2, 2]
    
    # p[9] is unused for now, could be used for another term or kept for future expansion
    # To avoid unused param issues, let's add a small constant offset or another term.
    # Let's add a Gaussian bump centered at 0 with learnable width? No, that adds non-linearity.
    # Let's just use p[9] as a global scale or offset? 
    # Actually, let's add another sinusoid or a different Chebyshev term?
    # Let's stick to the strategy: "single learnable-frequency sinusoidal term and a learnable exponential term".
    # That uses 5 + 2 + 2 = 9 params. We have 10.
    # Let's add a second exponential? Or a second sinusoid?
    # The directive says: "hybrid basis that combines a fixed, well-conditioned Chebyshev polynomial series (degrees 0–4) with a single learnable-frequency sinusoidal term (sin(w*x)) and a learnable exponential term (a*exp(b*x))"
    # This specifies 5 + 2 + 2 = 9 params. 
    # The interface requires 10 params. 
    # I will add a constant offset term using p[9] to ensure all params are used and the model is flexible.
    # Or better, extend Chebyshev to degree 5? 
    # "Chebyshev polynomial series (degrees 0–4)" implies 5 terms.
    # Let's use p[9] as an additional constant offset, or scale the whole thing?
    # Let's use p[9] as an amplitude for a second, fixed-frequency sinusoid? No, keep it simple.
    # Let's just add p[9] * 1.0 (constant offset) to the Chebyshev part? But c0 is already a constant.
    # It's redundant. 
    # Let's use p[9] to modulate the exponential width? No.
    # Let's add a term: p[9] * exp(-x**2 / 2.0) ? A Gaussian bump. This is common in symbolic regression.
    # It adds non-linearity but is bounded and smooth.
    
    # Scale x to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Linear combination of Chebyshev terms
    y_cheb = (c0 * T0 + 
              c1 * T1 + 
              c2 * T2 + 
              c3 * T3 + 
              c4 * T4)
    
    # Sinusoidal term
    y_sin = A * np.sin(w * x)
    
    # Exponential term
    # Clip C*x to prevent overflow in exp
    exp_arg = C * x
    exp_arg = np.clip(exp_arg, -50, 50)
    y_exp = B * np.exp(exp_arg)
    
    # Gaussian bump using p[9]
    # Centered at 0, width fixed at 1.0 for simplicity, amplitude learnable
    y_gauss = p[9] * np.exp(-0.5 * x**2)
    
    result = y_cheb + y_sin + y_exp + y_gauss
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
