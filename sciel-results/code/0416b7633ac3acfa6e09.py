"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a composite mechanistic-phenomenological form:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Logistic inhibition for biomass (b)
    - Linear interaction terms to capture coupling effects
    
    Parameters (10 total):
    p[0]: Amplitude (max growth rate scale)
    p[1]: Ks (Monod half-saturation constant)
    p[2]: Temp Gaussian width
    p[3]: Temp optimal value
    p[4]: pH Gaussian width
    p[5]: pH optimal value
    p[6]: Biomass inhibition coefficient
    p[7]: Offset
    p[8]: Temp-substrate interaction coefficient
    p[9]: pH-substrate interaction coefficient
    """
    p = params
    
    # Normalize inputs to reasonable ranges for stability
    # b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    b_norm = b / 5.0
    s_norm = s / 5.0
    temp_norm = temp / 40.0
    ph_norm = pH / 12.0
    
    # Monod kinetics for substrate
    monod_s = s / (p[1] + s)
    
    # Gaussian response for temperature
    temp_gauss = np.exp(-((temp - p[3]) ** 2) / (2.0 * p[2] ** 2))
    
    # Gaussian response for pH
    ph_gauss = np.exp(-((pH - p[5]) ** 2) / (2.0 * p[4] ** 2))
    
    # Logistic inhibition for biomass
    biomass_inhibit = 1.0 / (1.0 + np.exp(p[6] * (b - 2.0)))
    
    # Base growth rate
    base = p[0] * monod_s * temp_gauss * ph_gauss * biomass_inhibit
    
    # Interaction terms
    interaction = p[8] * temp_norm * monod_s + p[9] * ph_norm * monod_s
    
    # Final prediction
    return base + interaction + p[7]
# EVOLVE-BLOCK-END
