"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a modified Ramberg-Osgood model with temperature dependence.
    
    sigma = sigma0(T) + E(T) * eps * (1 + C(T) * ((eps^2 + delta^2) / (eps_y(T)^2 + delta^2))^(n(T)/2))
    
    Parameters:
    p[0]: E0 (Young's Modulus scale)
    p[1]: alpha_E (Temp coeff for E)
    p[2]: eps_y0 (Yield Strain scale)
    p[3]: alpha_y (Temp coeff for eps_y)
    p[4]: n0 (Hardening exponent scale)
    p[5]: alpha_n (Temp coeff for n)
    p[6]: C0 (Hardening coefficient scale)
    p[7]: alpha_C (Temp coeff for C)
    p[8]: sigma0_0 (Intercept scale)
    p[9]: sigma0_T (Intercept temp coeff)
    """
    E0 = params[0]
    alpha_E = params[1]
    eps_y0 = params[2]
    alpha_y = params[3]
    n0 = params[4]
    alpha_n = params[5]
    C0 = params[6]
    alpha_C = params[7]
    sigma0_0 = params[8]
    sigma0_T = params[9]
    
    # Temperature dependent parameters
    E = E0 * (1.0 + alpha_E * temp)
    eps_y = eps_y0 * (1.0 + alpha_y * temp)
    n = n0 + alpha_n * temp
    C = C0 * (1.0 + alpha_C * temp)
    sigma0 = sigma0_0 + sigma0_T * temp
    
    # Smooth power term
    delta = 1e-6
    eps_sq = strain * strain + delta * delta
    eps_y_sq = eps_y * eps_y + delta * delta
    
    # Ensure base is positive
    ratio = eps_sq / eps_y_sq
    
    # Handle potential numerical issues with power
    # If n is very large or small, power might be unstable.
    # Clip n to a reasonable range? No, let the fitter decide, but ensure finite output.
    
    # term = ratio ** (n / 2.0)
    # If n is negative, ratio^(n/2) is large when ratio is small.
    # We assume n > 0 for physical hardening.
    
    # To avoid NaNs from 0**negative or inf, we clip ratio
    ratio_clipped = np.clip(ratio, 1e-12, 1e12)
    
    term = np.power(ratio_clipped, n / 2.0)
    
    # Combine
    stress = sigma0 + E * strain * (1.0 + C * term)
    
    return stress
# EVOLVE-BLOCK-END
