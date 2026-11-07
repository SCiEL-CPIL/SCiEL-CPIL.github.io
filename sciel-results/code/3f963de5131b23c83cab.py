"""E. coli growth rate model with enhanced polynomial and biological basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled using a Fourier basis for temperature and pH
    to capture non-Gaussian environmental responses, combined with Monod
    substrate kinetics and density inhibition.
    
    Basis functions:
      - Constant
      - Monod substrate saturation
      - Density inhibition
      - First and second harmonic Fourier terms for Temperature
      - First harmonic Fourier terms for pH
      - Interaction between substrate and temperature
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological/physical scales
    Ks = 5.0        # Substrate half-saturation
    Kd = 1.0        # Density inhibition constant
    
    # Core biological terms
    monod_s = s / (Ks + s)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Fourier terms for Temperature
    # Using a period of 50 degrees to capture broad trends
    omega_T = 2.0 * np.pi / 50.0
    sin_T1 = np.sin(omega_T * temp)
    cos_T1 = np.cos(omega_T * temp)
    sin_T2 = np.sin(2.0 * omega_T * temp)
    cos_T2 = np.cos(2.0 * omega_T * temp)
    
    # Fourier terms for pH
    # Using a period of 5 pH units to capture local variations
    omega_pH = 2.0 * np.pi / 5.0
    sin_pH1 = np.sin(omega_pH * pH)
    cos_pH1 = np.cos(omega_pH * pH)
    
    # Interaction term
    interaction = monod_s * cos_T1
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * 1.0 +
        p1 * monod_s +
        p2 * density_inhib +
        p3 * sin_T1 +
        p4 * cos_T1 +
        p5 * sin_T2 +
        p6 * cos_T2 +
        p7 * sin_pH1 +
        p8 * cos_pH1 +
        p9 * interaction
    )
    
    return prediction
# EVOLVE-BLOCK-END
