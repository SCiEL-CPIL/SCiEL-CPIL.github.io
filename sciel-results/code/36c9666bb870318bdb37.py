"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with Monod kinetics, temperature/pH optima,
    and density inhibition.
    
    params (10 values):
        p[0]: log(max growth rate scale)
        p[1]: log(Monod half-saturation Ks)
        p[2]: optimal temperature
        p[3]: log(temperature width sigma_T)
        p[4]: optimal pH
        p[5]: log(pH width sigma_pH)
        p[6]: log(density inhibition constant Kb)
        p[7]: density inhibition exponent
        p[8]: linear density coefficient
        p[9]: offset
    """
    # Extract and transform parameters
    mu_max = np.exp(params[0])
    Ks = np.exp(params[1])
    T_opt = params[2]
    sigma_T = np.exp(params[3])
    pH_opt = params[4]
    sigma_pH = np.exp(params[5])
    Kb = np.exp(params[6])
    n_inh = params[7]
    c_b = params[8]
    offset = params[9]
    
    # Monod substrate term
    monod = s / (Ks + s)
    
    # Temperature response (Gaussian)
    temp_resp = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH response (Gaussian)
    ph_resp = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition: reduces growth at high density
    # Use a Haldane-like term: 1 / (1 + (b/Kb)^n)
    density_inh = 1.0 / (1.0 + (b / Kb) ** n_inh)
    
    # Linear density term (small effect)
    density_linear = 1.0 + c_b * b
    
    # Combine all terms
    growth = mu_max * monod * temp_resp * ph_resp * density_inh * density_linear + offset
    
    return growth
# EVOLVE-BLOCK-END
