"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a mechanistic approach.
    
    This model combines Monod kinetics for substrate limitation, Gaussian
    responses for temperature and pH optima, and a logistic-type inhibition
    for population density. It includes linear correction terms to capture
    any residual trends not explained by the primary mechanistic components.
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Log max growth rate (exp(p[0]))
        p[1]: Log Ks (half-saturation constant, exp(p[1]))
        p[2]: Optimal temperature
        p[3]: Log temperature width (exp(p[3]))
        p[4]: Optimal pH
        p[5]: Log pH width (exp(p[5]))
        p[6]: Log density constant (exp(p[6]))
        p[7]: Linear coefficient for s
        p[8]: Linear coefficient for temp
        p[9]: Linear coefficient for pH
    """
    p = params
    
    # Extract parameters and apply transformations for numerical stability
    # Using exponential for positive-only parameters (widths, constants)
    # allows BFGS to optimize in log-space, ensuring positivity.
    p_max = np.exp(p[0])
    Ks = np.exp(p[1])
    T_opt = p[2]
    T_width = np.exp(p[3])
    pH_opt = p[4]
    pH_width = np.exp(p[5])
    b_const = np.exp(p[6])
    
    # 1. Monod kinetics for substrate (s)
    # y_s = s / (Ks + s)
    # Ks is positive, s is positive, so this is stable.
    monod_s = s / (Ks + s)
    
    # 2. Gaussian response for temperature (temp)
    # y_t = exp( - (temp - T_opt)^2 / T_width^2 )
    # T_width is positive.
    T_term = np.exp(-((temp - T_opt) ** 2) / (T_width ** 2))
    
    # 3. Gaussian response for pH
    # y_p = exp( - (pH - pH_opt)^2 / pH_width^2 )
    pH_term = np.exp(-((pH - pH_opt) ** 2) / (pH_width ** 2))
    
    # 4. Density inhibition (b)
    # y_b = 1 / (1 + (b / b_const)^2)
    # This provides a smooth decay as density increases.
    b_term = 1.0 / (1.0 + (b / b_const) ** 2)
    
    # Main mechanistic term: Product of all factors
    y_mech = p_max * monod_s * T_term * pH_term * b_term
    
    # 5. Linear correction terms
    # These allow the model to fit linear trends that the multiplicative
    # mechanistic model might miss, improving flexibility.
    y_lin = p[7] * s + p[8] * temp + p[9] * pH
    
    # Final prediction
    return y_mech + y_lin
# EVOLVE-BLOCK-END
