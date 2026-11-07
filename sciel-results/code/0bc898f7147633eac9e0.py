"""E. coli growth rate model with piecewise-linear growth plateau."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a piecewise-linear growth plateau.
    
    Key structural features:
      - Hard zero-growth boundaries for suboptimal temperature and pH
        using np.clip to enforce strict biologically-impossible regime cutoffs
      - Rational inhibition term for substrate (Monod-like) and density
      - Piecewise-linear response within viable range
      - 10 linear coefficients for BFGS optimization
    
    This creates a hard-constrained landscape that prevents BFGS from
    fitting growth in biologically impossible regimes.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological boundaries (based on E. coli physiology)
    T_min = 25.0    # Minimum viable temperature
    T_max = 45.0    # Maximum viable temperature
    T_opt = 37.0    # Optimal temperature
    pH_min = 5.0    # Minimum viable pH
    pH_max = 9.0    # Maximum viable pH
    pH_opt = 7.0    # Optimal pH
    Ks = 5.0        # Substrate half-saturation
    Kd = 2.0        # Density inhibition constant
    
    # Piecewise-linear temperature response with hard boundaries
    # Zero below T_min, linear ramp up to T_opt, plateau, linear ramp down to T_max, zero above
    temp_below = np.clip((temp - T_min) / (T_opt - T_min), 0.0, 1.0)
    temp_above = np.clip((T_max - temp) / (T_max - T_opt), 0.0, 1.0)
    temp_plateau = np.minimum(temp_below, temp_above)
    
    # Piecewise-linear pH response with hard boundaries
    ph_below = np.clip((pH - pH_min) / (pH_opt - pH_min), 0.0, 1.0)
    ph_above = np.clip((pH_max - pH) / (pH_max - pH_opt), 0.0, 1.0)
    ph_plateau = np.minimum(ph_below, ph_above)
    
    # Combined environmental viability (product of temp and pH plateaus)
    env_viability = temp_plateau * ph_plateau
    
    # Substrate saturation (Monod-like rational term)
    monod_s = s / (Ks + s)
    
    # Density inhibition (rational term)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Normalized inputs for additional polynomial flexibility
    b_norm = np.clip((b - 1.0) / (5.0 - 1.0), -1.0, 1.0)
    s_norm = np.clip((s - 1.0) / (40.0 - 1.0), -1.0, 1.0)
    temp_norm = np.clip((temp - T_opt) / (T_opt - T_min), -1.0, 1.0)
    ph_norm = np.clip((pH - pH_opt) / (pH_opt - pH_min), -1.0, 1.0)
    
    # Additional basis terms for flexibility within viable range
    temp_quad = temp_norm ** 2
    ph_quad = ph_norm ** 2
    temp_ph_cross = temp_norm * ph_norm
    s_temp_cross = s_norm * temp_norm
    s_ph_cross = s_norm * ph_norm
    
    # Combine all basis functions with fitted parameters
    # Primary term: full environmental constraint
    primary = p0 * monod_s * env_viability * density_inhib
    
    # Secondary terms: partial environmental constraints
    temp_only = p1 * monod_s * temp_plateau * density_inhib
    ph_only = p2 * monod_s * ph_plateau * density_inhib
    env_no_density = p3 * monod_s * env_viability
    
    # Substrate-only and environmental-only terms
    substrate_only = p4 * monod_s
    temp_response = p5 * temp_plateau
    ph_response = p6 * ph_plateau
    
    # Polynomial correction terms (active only within viable range)
    temp_correction = p7 * temp_quad * env_viability
    ph_correction = p8 * ph_quad * env_viability
    cross_correction = p9 * temp_ph_cross * env_viability
    
    prediction = (
        primary +
        temp_only +
        ph_only +
        env_no_density +
        substrate_only +
        temp_response +
        ph_response +
        temp_correction +
        ph_correction +
        cross_correction
    )
    
    return prediction
# EVOLVE-BLOCK-END
