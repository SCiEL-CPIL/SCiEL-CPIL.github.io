import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Physically-Motivated Non-Linear Composite Model (Ramberg-Osgood Variant).
    
    Unlike the linear basis expansion, this model uses a non-linear functional form
    where parameters interact multiplicatively and exponentially. This is a different
    design family: it encodes the physics of elastic-plastic deformation.
    
    The form is:
    sigma = E(T) * [ eps + 0.002 * (sigma_y(T) / E(T))^(1/n) * (sigma / E(T))^n ]
    
    However, to keep it explicit and avoid implicit equations which are hard for BFGS 
    to solve directly in a single forward pass without fixed-point iteration (which 
    might be too slow or unstable), we use an explicit approximation of the 
    Ramberg-Osgood relation:
    
    sigma = E(T) * eps + K(T) * eps^n(T)
    
    Where:
    E(T) = p[0] + p[1] * (t - 0.5)  # Linear temperature dependence of modulus
    K(T) = p[2] + p[3] * (t - 0.5)  # Linear temperature dependence of hardening coeff
    n(T) = p[4] + p[5] * (t - 0.5)  # Linear temperature dependence of hardening exponent
    
    To ensure robustness and cover the full range including potential drop in stress 
    (if any, though Al usually hardens until necking which might be out of range or 
    represented by saturation), we add a saturation term or a second hardening term.
    
    Let's use a more flexible form:
    sigma = E(T) * eps * (1 - p[6] * exp(-p[7] * eps)) + p[8] * (1 - exp(-p[9] * eps))
    
    This is a sum of:
    1. An elastic term that softens slightly or hardens depending on p[6].
       Actually, standard elastic is linear. Let's stick to:
       sigma = E(T) * eps + Hardening(eps, T)
       
    Hardening(eps, T) = A(T) * eps^B(T) + C(T) * exp(-D(T) * eps)
    
    Parameters:
    p[0]: E_base
    p[1]: E_temp_coeff
    p[2]: A_base (hardening amplitude)
    p[3]: A_temp_coeff
    p[4]: B_base (hardening exponent, usually 0.1-0.5)
    p[5]: B_temp_coeff
    p[6]: C_base (exponential transient amplitude)
    p[7]: C_temp_coeff
    p[8]: D_base (exponential decay rate)
    p[9]: D_temp_coeff
    
    This form is non-linear in params (due to eps^B and exp(-D*eps)), making the 
    landscape non-convex. However, it is structurally closer to the data generation 
    process for metallic stress-strain curves than a generic polynomial/Gaussian sum.
    """
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    p = params
    
    # Ensure params are 1D and float
    p = np.asarray(p, dtype=np.float64).flatten()
    
    # If any param is non-finite, return zeros to avoid NaNs in MSE
    if not np.all(np.isfinite(p)):
        return np.zeros_like(s)
        
    # Temperature normalization: t is in [0.06667, 1]. Center at 0.5 for stability.
    t_norm = t - 0.5
    
    # Elastic Modulus
    E = p[0] + p[1] * t_norm
    
    # Hardening Term 1: Power Law A * eps^B
    # Ensure base of power is non-negative. Strain can be negative in training? 
    # Range: [-0.001911, 0.9636]. 
    # Negative strain (compression) might exist. 
    # For negative strain, physical models differ. 
    # To be safe and differentiable, use sign(s) * |s|^B or just s * |s|^(B-1).
    # Let's use a smooth odd-like function for the hardening part if strain is negative.
    # However, typical tensile tests are positive. The negative values are small.
    # Let's use np.sign(s) * np.power(np.abs(s), B) to handle negative strain smoothly.
    
    A = p[2] + p[3] * t_norm
    B = p[4] + p[5] * t_norm
    
    # Clip B to a reasonable range to avoid extreme values during optimization
    # BFGS might wander. If B is huge, s^B is 0 for s<1 and 1 for s>1.
    # If B is negative, s^B blows up at 0.
    # Let's constrain B to [0.1, 5.0] via tanh mapping? No, params are raw.
    # We must rely on the landscape. But to prevent NaNs from 0^negative, 
    # we add a small epsilon to abs(s).
    s_abs = np.abs(s) + 1e-9
    hardening_power = A * np.sign(s) * np.power(s_abs, B)
    
    # Hardening Term 2: Exponential Transient C * (1 - exp(-D * s))
    # This captures the rapid initial hardening or yield point behavior.
    C = p[6] + p[7] * t_norm
    D = p[8] + p[9] * t_norm
    
    # For negative s, exp(-D*s) might blow up if D*s is large negative.
    # Clip the argument to exp to prevent overflow.
    arg_exp = -D * s
    # If D is negative and s is positive, arg is positive -> overflow.
    # If D is positive and s is negative, arg is positive -> overflow.
    # Clip to [-700, 700]
    arg_exp_clipped = np.clip(arg_exp, -700, 700)
    hardening_exp = C * (1 - np.exp(arg_exp_clipped))
    
    # Total Stress
    stress = E * s + hardening_power + hardening_exp
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
