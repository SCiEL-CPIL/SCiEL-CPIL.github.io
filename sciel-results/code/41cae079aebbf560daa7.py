"""Modified Voce hardening model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = sigma_0(T) + sigma_s(T) * (1 - exp(-strain / epsilon_c(T)))
    
    This model captures:
    1. Initial yield stress (sigma_0)
    2. Saturation stress (sigma_s)
    3. Hardening rate/strain scale (epsilon_c)
    All modulated linearly by temperature.
    
    Inputs:
    strain: np.ndarray
    temp: np.ndarray
    params: np.ndarray of length 10
    """
    # Unpack 10 parameters
    # p0, p1: sigma_0 = p0 + p1 * temp
    # p2, p3: sigma_s = p2 + p3 * temp
    # p4, p5: epsilon_c = p4 + p5 * temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate temperature-dependent coefficients
    sigma_0 = p0 + p1 * temp
    sigma_s = p2 + p3 * temp
    epsilon_c = p4 + p5 * temp
    
    # To prevent division by zero or negative exponent issues during optimization,
    # we can add a small epsilon or rely on the optimizer avoiding singularities.
    # However, for robustness in BFGS which may probe negative values,
    # we can use a safe division. But standard Voce is simple.
    # Let's add a small constant to denominator if needed, but usually
    # the optimizer will find positive epsilon_c.
    # To be safe against division by zero during the fit:
    denom = epsilon_c
    # If denom is close to 0, exp(-strain/denom) can be unstable.
    # We can clip denom to a small positive value if we assume hardening is positive.
    # But the model should allow the optimizer to decide.
    # A safer approach for the exponent:
    # term = 1 - exp(-strain / epsilon_c)
    
    # Let's use a formulation that is numerically stable.
    # If epsilon_c is very small, exp(-large) -> 0, term -> 1.
    # If epsilon_c is large, exp(-small) -> 1, term -> 0.
    
    # We will ignore p6-p9 for now as 6 params are sufficient for this form,
    # but we must use all 10 to match the interface? 
    # The prompt says "up to 10 fitted constants". 
    # The evaluator fits an array of length 10. 
    # If I don't use p6-p9, they will be ignored by the gradient (zero gradient) 
    # and stay at 1.0 (initial value). This is fine.
    
    # Calculate the hardening term
    # Clip strain to avoid extreme values if optimizer goes wild, though input range is fixed.
    # strain is in [-0.0019, 0.96].
    
    # Safe division
    # If epsilon_c is 0, we have a problem. 
    # Let's add a tiny epsilon to the denominator to prevent inf/nan during BFGS steps.
    # However, if the true solution has epsilon_c near 0, this biases it.
    # Given the physical context, epsilon_c should be > 0.
    # We can use np.where to handle signs, but let's trust the optimizer 
    # to stay in a reasonable range, or use a softplus-like transform?
    # No, keep it simple. The initial value is 1.0.
    
    # To be extra safe against numerical issues in exp:
    # exp(-x) where x = strain / epsilon_c
    # If epsilon_c is negative, x is negative (for positive strain), exp(-x) = exp(positive) -> overflow.
    # To prevent overflow, we can clip the exponent argument.
    arg = strain / epsilon_c
    # Clip arg to [-100, 100] to prevent exp overflow/underflow issues
    # exp(100) is ~ 2.7e43, which is fine for float64.
    # exp(-100) is ~ 3.7e-44, fine.
    arg_clipped = np.clip(arg, -100, 100)
    
    hardening_term = 1.0 - np.exp(-arg_clipped)
    
    stress = sigma_0 + sigma_s * hardening_term
    
    return stress
# EVOLVE-BLOCK-END
