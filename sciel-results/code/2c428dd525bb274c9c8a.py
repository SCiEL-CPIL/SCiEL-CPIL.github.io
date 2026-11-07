"""E. coli growth rate model with piecewise-linear growth plateau and rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate with hard zero-growth boundaries via piecewise-linear plateaus.
    
    Uses np.clip to enforce strict zero growth outside optimal temperature and pH ranges,
    combined with Monod substrate saturation and density inhibition. This creates a
    hard-constrained landscape that prevents BFGS from fitting growth in biologically
    impossible regimes.
    
    Basis functions:
      - Monod substrate saturation: s / (Ks + s)
      - Piecewise-linear temperature plateau (zero outside [T_min, T_max])
      - Piecewise-linear pH plateau (zero outside [pH_min, pH_max])
      - Density inhibition: 1 / (1 + Kd * b)
      - Interaction terms
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological parameters
    Ks = 5.0        # Substrate half-saturation
    T_opt = 37.0    # Optimal temperature
    T_width = 5.0   # Half-width of linear plateau
    pH_opt = 7.0    # Optimal pH
    pH_width = 1.5  # Half-width of linear plateau
    Kd = 1.0        # Density inhibition constant
    
    # Piecewise-linear temperature plateau: linear ramp to 0 outside [T_opt-T_width, T_opt+T_width]
    T_min = T_opt - T_width
    T_max = T_opt + T_width
    # Linear rise from 0 at T_min to 1 at T_opt, linear fall from 1 at T_opt to 0 at T_max
    temp_ramp_up = np.clip((temp - T_min) / (T_opt - T_min), 0.0, 1.0)
    temp_ramp_down = np.clip((T_max - temp) / (T_max - T_opt), 0.0, 1.0)
    temp_plateau = np.minimum(temp_ramp_up, temp_ramp_down)
    
    # Piecewise-linear pH plateau
    pH_min = pH_opt - pH_width
    pH_max = pH_opt + pH_width
    ph_ramp_up = np.clip((pH - pH_min) / (pH_opt - pH_min), 0.0, 1.0)
    ph_ramp_down = np.clip((pH_max - pH) / (pH_max - pH_opt), 0.0, 1.0)
    ph_plateau = np.minimum(ph_ramp_up, ph_ramp_down)
    
    # Monod substrate saturation
    monod_s = s / (Ks + s)
    
    # Density inhibition
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Core interaction term: all factors multiplied
    core = monod_s * temp_plateau * ph_plateau * density_inhib
    
    # Combine basis functions with fitted parameters
    prediction = (
        p0 * core +
        p1 * monod_s * temp_plateau +
        p2 * monod_s * ph_plateau +
        p3 * temp_plateau * ph_plateau +
        p4 * monod_s +
        p5 * temp_plateau +
        p6 * ph_plateau +
        p7 * density_inhib * monod_s +
        p8 * monod_s * temp_plateau * ph_plateau +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
