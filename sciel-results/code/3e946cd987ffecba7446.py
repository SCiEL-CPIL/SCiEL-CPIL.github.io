"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with biological priors.
    
    Model structure:
    - Monod kinetics for substrate limitation
    - Gaussian temperature response (optimal ~37C)
    - Gaussian pH response (optimal ~7)
    - Density-dependent effects (inhibition at high density)
    
    params (length 10):
    p[0]: max growth rate
    p[1]: Monod constant Ks
    p[2]: optimal temperature
    p[3]: temperature width (sigma)
    p[4]: optimal pH
    p[5]: pH width (sigma)
    p[6]: density inhibition coefficient
    p[7]: density inhibition half-saturation
    p[8]: baseline offset
    p[9]: Hill coefficient for substrate (flexibility)
    """
    p = params
    
    # Substrate term: Monod-like with flexible Hill coefficient
    # s / (Ks + s^h) to allow for more complex saturation
    h = 1.0 + 0.5 * np.tanh(p[9])  # Hill coefficient in [0.5, 1.5]
    Ks = np.abs(p[1]) + 1e-6
    substrate_term = s / (Ks + np.power(np.abs(s), h))
    
    # Temperature term: Gaussian centered at optimal temp
    # Normalize to [0,1] range for stability
    temp_opt = p[2]
    temp_width = np.abs(p[3]) + 1.0
    temp_term = np.exp(-0.5 * ((temp - temp_opt) / temp_width) ** 2)
    
    # pH term: Gaussian centered at optimal pH
    ph_opt = p[4]
    ph_width = np.abs(p[5]) + 1.0
    ph_term = np.exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2)
    
    # Density term: Inhibition at high density
    # Use a form that reduces growth at high b
    b_inhib = 1.0 / (1.0 + (np.abs(b) / (np.abs(p[7]) + 1e-6)) ** np.abs(p[6]) + 0.1)
    
    # Combine all terms
    growth = p[0] * substrate_term * temp_term * ph_term * b_inhib + p[8]
    
    return growth
# EVOLVE-BLOCK-END
