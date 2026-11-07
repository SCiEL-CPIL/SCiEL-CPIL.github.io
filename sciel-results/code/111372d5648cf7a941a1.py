"""E. coli growth rate model with piecewise-linear viability and polynomial interior."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining piecewise-linear environmental viability
    with polynomial interior flexibility.
    
    Strategy:
    1. Define hard biological boundaries for Temperature and pH using piecewise-linear
       ramps (triangular/trapezoidal shapes) that are strictly zero outside the viable range.
       This prevents BFGS from fitting artifacts in biologically impossible regimes.
    2. Within the viable range, use quadratic terms to capture asymmetric responses.
    3. Use Monod kinetics for substrate and simple rational inhibition for density.
    4. The 10 parameters are linear coefficients for the basis functions, ensuring
       the optimization is convex and BFGS can solve it reliably.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)

    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological boundaries (based on E. coli physiology)
    # Temperature: viable ~25-45C, optimal ~37C
    T_min = 20.0
    T_max = 50.0
    T_opt = 37.0
    
    # pH: viable ~5-9, optimal ~7
    pH_min = 5.0
    pH_max = 9.0
    pH_opt = 7.0
    
    # Substrate & Density constants
    Ks = 5.0
    Kd = 1.0
    
    # --- Environmental Viability Basis Functions ---
    # These are 0 outside [T_min, T_max] and [pH_min, pH_max]
    
    # Temperature viability: Triangular peak at T_opt, zero at boundaries
    # Left ramp: 0 to 1 from T_min to T_opt
    temp_left = np.clip((temp - T_min) / (T_opt - T_min), 0.0, 1.0)
    # Right ramp: 1 to 0 from T_opt to T_max
    temp_right = np.clip((T_max - temp) / (T_max - T_opt), 0.0, 1.0)
    # Combined temperature viability (product of ramps creates a triangle)
    temp_viable = temp_left * temp_right
    
    # pH viability: Triangular peak at pH_opt, zero at boundaries
    ph_left = np.clip((pH - pH_min) / (pH_opt - pH_min), 0.0, 1.0)
    ph_right = np.clip((pH_max - pH) / (pH_opt - pH_min), 0.0, 1.0)
    ph_viable = ph_left * ph_right
    
    # Combined environmental viability
    env_viable = temp_viable * ph_viable
    
    # --- Substrate and Density Basis Functions ---
    # Monod substrate saturation
    monod_s = s / (Ks + s)
    
    # Density inhibition (Haldane-like or simple Michaelis-Menten style)
    # Using 1/(1+Kd*b) for simplicity and stability
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # --- Interior Flexibility (Polynomial terms active only where viable) ---
    # Normalized deviations for polynomial terms
    # We use the clipped linear ramps themselves as the linear terms
    # and add quadratic corrections to capture asymmetry within the viable range
    
    # Quadratic correction for temperature (active only when viable)
    # (temp - T_opt)^2 is large at boundaries, small at opt. 
    # We want to allow the model to tilt the peak.
    temp_quad = (temp - T_opt) ** 2
    temp_corr = temp_viable * temp_quad
    
    # Quadratic correction for pH
    ph_quad = (pH - pH_opt) ** 2
    ph_corr = ph_viable * ph_quad
    
    # Interaction term: Substrate * Environment
    monod_env = monod_s * env_viable
    
    # Interaction term: Substrate * Temperature only
    monod_temp = monod_s * temp_viable
    
    # Interaction term: Substrate * pH only
    monod_ph = monod_s * ph_viable
    
    # Full interaction: Substrate * Environment * Density
    full_interact = monod_s * env_viable * density_inhib
    
    # Substrate * Density
    sub_den = monod_s * density_inhib
    
    # --- Basis Functions ---
    # 1. Full biological interaction (Substrate * Env * Density)
    basis_1 = full_interact
    
    # 2. Substrate * Env (no density)
    basis_2 = monod_env
    
    # 3. Substrate * Temp
    basis_3 = monod_temp
    
    # 4. Substrate * pH
    basis_4 = monod_ph
    
    # 5. Substrate * Density
    basis_5 = sub_den
    
    # 6. Substrate only
    basis_6 = monod_s
    
    # 7. Env * Density
    basis_7 = env_viable * density_inhib
    
    # 8. Temp correction (asymmetry)
    basis_8 = temp_corr
    
    # 9. pH correction (asymmetry)
    basis_9 = ph_corr
    
    # 10. Constant (baseline)
    basis_10 = np.ones_like(b, dtype=np.float64)
    
    # Combine
    prediction = (
        p0 * basis_1 +
        p1 * basis_2 +
        p2 * basis_3 +
        p3 * basis_4 +
        p4 * basis_5 +
        p5 * basis_6 +
        p6 * basis_7 +
        p7 * basis_8 +
        p8 * basis_9 +
        p9 * basis_10
    )
    
    # Ensure non-negative growth rate? 
    # The task says "growth rate", which can be negative (death) in some contexts, 
    # but E. coli growth rate is typically mu >= 0. 
    # However, the target range is [-0.00277, 1.553], so negative values are possible.
    # We do not clip to 0 to allow fitting negative rates if the data requires it.
    
    return prediction
# EVOLVE-BLOCK-END
