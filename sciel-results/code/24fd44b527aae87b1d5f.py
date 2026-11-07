"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model based on Monod kinetics, temperature and pH optima.
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Max growth rate scale
        p[1]: Log of Monod half-saturation constant (Ks)
        p[2]: Optimal temperature
        p[3]: Log of temperature width (sigma_T)
        p[4]: Optimal pH
        p[5]: Log of pH width (sigma_pH)
        p[6]: Log of density inhibition constant (Kb)
        p[7]: Density inhibition exponent
        p[8]: Linear coefficient for density
        p[9]: Bias
    """
    p = params
    
    # Extract parameters
    mu_max = p[0]
    log_Ks = p[1]
    T_opt = p[2]
    log_sig_T = p[3]
    pH_opt = p[4]
    log_sig_pH = p[5]
    log_Kb = p[6]
    exp_b = p[7]
    lin_b = p[8]
    bias = p[9]
    
    # Calculate terms
    # Substrate term: Monod kinetics
    # Ks = exp(log_Ks) ensures Ks > 0
    Ks = np.exp(log_Ks)
    # Avoid division by zero if Ks is extremely small and s is 0, but s >= 1
    monod_s = s / (s + Ks)
    
    # Temperature term: Gaussian
    # sig_T = exp(log_sig_T) ensures sig_T > 0
    sig_T = np.exp(log_sig_T)
    # Prevent overflow in exponent calculation if sig_T is tiny
    # (temp - T_opt)^2 can be large. 
    temp_diff_sq = (temp - T_opt) ** 2
    # If sig_T is very small, sig_T**2 is very small, division is huge.
    # exp(-huge) -> 0. This is desired behavior (sharp peak).
    # To avoid runtime warnings for overflow in square or division, we can clip or just let it go to 0.
    # NumPy handles inf in exp gracefully (exp(-inf) = 0).
    temp_factor = np.exp(-temp_diff_sq / (2 * sig_T**2))
    
    # pH term: Gaussian
    sig_pH = np.exp(log_sig_pH)
    pH_diff_sq = (pH - pH_opt) ** 2
    pH_factor = np.exp(-pH_diff_sq / (2 * sig_pH**2))
    
    # Density term: Inhibition
    # Kb = exp(log_Kb) ensures Kb > 0
    Kb = np.exp(log_Kb)
    # b is positive. exp_b is a parameter.
    # If exp_b is large and b is large, b**exp_b can overflow.
    # b is in [1, 5]. If exp_b is 10, 5**10 is ~1e7. Safe.
    # If exp_b is 100, 5**100 is huge. 
    # We can clip the exponent result or use a safer form.
    # Let's assume reasonable bounds for exp_b during optimization.
    # To be safe, we can clip the power term.
    b_power = np.power(b, exp_b)
    # If b_power is inf, 1/(1+inf) is 0.
    density_inhibition = 1.0 / (1.0 + Kb * b_power)
    
    # Combine
    # The linear term lin_b * b allows for non-saturating effects if the inhibition model is insufficient
    # Or it might just be a small correction.
    y = mu_max * monod_s * temp_factor * pH_factor * density_inhibition + lin_b * b + bias
    
    return y
# EVOLVE-BLOCK-END
