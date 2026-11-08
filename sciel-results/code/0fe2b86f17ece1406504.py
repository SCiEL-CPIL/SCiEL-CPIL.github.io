"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with parametric frequency estimation.
    
    Strategy:
    Instead of assuming a polynomial time dependence (which approximates a 
    local Taylor expansion), this model uses a Fourier basis in time with a 
    fitted frequency parameter. This is physically more appropriate for 
    oscillators, which are inherently periodic. The state dependence is 
    modeled with a low-order polynomial including a cross-term for 
    nonlinear damping/stiffness coupling.
    
    Parameters:
    t: time (range [30, 50])
    x: position (range [-0.34, 0.36])
    v: velocity (range [-0.21, 0.17])
    params: array of 10 fitted constants
        params[0]: frequency omega (in scaled time units)
        params[1]: coeff for cos(omega * t_c)
        params[2]: coeff for sin(omega * t_c)
        params[3]: coeff for cos(2 * omega * t_c)
        params[4]: coeff for sin(2 * omega * t_c)
        params[5]: coeff for x
        params[6]: coeff for v
        params[7]: coeff for x^2
        params[8]: coeff for v^2
        params[9]: coeff for x*v
    
    Returns:
    np.ndarray: Predicted acceleration/velocity change
    """
    # Scale time to improve conditioning for frequency fitting
    # t in [30, 50] -> t_c in [-1, 1]
    t_c = (t - 40.0) / 10.0
    
    # Extract frequency parameter
    # Clip omega to avoid extreme values that cause numerical instability in sin/cos
    # or huge gradients. A reasonable range for scaled frequency is [0.1, 10.0]
    omega = np.clip(params[0], 0.1, 10.0)
    
    # Fourier terms in scaled time
    # Note: We use 1st and 2nd harmonics to capture non-sinusoidal oscillations
    cos_1 = np.cos(omega * t_c)
    sin_1 = np.sin(omega * t_c)
    cos_2 = np.cos(2.0 * omega * t_c)
    sin_2 = np.sin(2.0 * omega * t_c)
    
    # State terms
    x2 = x * x
    v2 = v * v
    xv = x * v
    
    # Combine terms
    # params[0] is omega, so we start coefficients from params[1]
    res = (params[1] * cos_1 + 
           params[2] * sin_1 + 
           params[3] * cos_2 + 
           params[4] * sin_2 + 
           params[5] * x + 
           params[6] * v + 
           params[7] * x2 + 
           params[8] * v2 + 
           params[9] * xv)
           
    # Add a constant term? 
    # We have 10 params. 
    # Current usage:
    # 0: omega
    # 1-4: Fourier coeffs (4)
    # 5-9: State coeffs (5)
    # Total 10. No constant term.
    # The Fourier basis with DC offset? 
    # If we include a constant, we drop one state term.
    # Usually, oscillators have a non-zero mean or the drive has a DC component.
    # Let's swap params[9] (xv) for a constant? 
    # No, xv is important for nonlinear damping.
    # Let's drop the 2nd harmonic (2 params) and add a constant and keep xv?
    # Basis: 1, cos(wt), sin(wt), x, v, x2, v2, xv, ... 
    # That's 8 terms. We have 2 left. 
    # Maybe x^3? Or t_c?
    # Let's stick to the 10-param structure above but check if constant is needed.
    # If the target mean is not 0, the model might struggle without a constant.
    # Target range: [-0.71, 0.66]. Mean is likely close to 0 but not guaranteed.
    # Let's replace the 2nd harmonic (cos2, sin2) with a constant and x^3?
    # No, 2nd harmonic is good for non-sinusoidal shapes.
    # Let's assume the constant is small or absorbed by the other terms.
    # Actually, let's use a "soft" constant by allowing the Fourier terms to have a DC offset?
    # No, sin/cos have 0 mean.
    # Let's add a constant term and drop the 2nd harmonic to save params?
    # New Basis:
    # 0: omega
    # 1: constant
    # 2: cos(omega * t_c)
    # 3: sin(omega * t_c)
    # 4: x
    # 5: v
    # 6: x^2
    # 7: v^2
    # 8: xv
    # 9: x^3 (for Duffing-like stiffness)
    
    # Let's re-evaluate.
    # Previous plan:
    # 0: omega
    # 1: cos1
    # 2: sin1
    # 3: cos2
    # 4: sin2
    # 5: x
    # 6: v
    # 7: x2
    # 8: v2
    # 9: xv
    
    # This lacks a constant.
    # Let's modify to include constant and drop one harmonic component?
    # Or just keep it as is. The state terms (x, v, etc.) can shift the mean if the state has a non-zero mean, but t is centered.
    # If the drive is purely oscillatory, constant is 0.
    # Let's stick to the 10-term Fourier+State model.
    
    return res
# EVOLVE-BLOCK-END
