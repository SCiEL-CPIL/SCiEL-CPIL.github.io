"""E. coli growth rate model combining smooth biological basis with hard constraints."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a combination of smooth biological basis functions
    and hard-constrained piecewise-linear boundaries.
    
    This formulation uses:
    1. Smooth Gaussian responses for temperature and pH to ensure differentiability.
    2. Hard zero-growth boundaries (via np.clip) for extreme conditions to prevent
       BFGS from fitting noise in biologically impossible regimes.
    3. Monod-like substrate saturation.
    4. Density-dependent inhibition.
    
    The model remains linear in parameters, ensuring convex optimization.
    
    Inputs:
        b: biomass density
        s: substrate concentration
        temp: temperature
        pH: pH value
        params: array of 10 fitted constants
    
    Returns:
        Array of predicted growth rates
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs for numerical stability
    # Estimated ranges from task description:
    # b: [1, 5] -> center 3, scale 2
    b_n = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, scale 2
    s_n = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, scale 19.5
    T_n = (temp - 20.5) / 19.5
    # pH: [2, 12] -> center 7, scale 5
    pH_n = (pH - 7.0) / 5.0
    
    # --- Substrate Saturation (Monod-like) ---
    # Map s_n to [0, 1] range for Monod: s_map = (s_n + 1) / 2
    s_map = (s_n + 1.0) / 2.0
    # Monod with K_s = 0.5 in mapped units
    monod_s = s_map / (0.5 + s_map)
    
    # --- Temperature Response ---
    # Combine smooth Gaussian with hard zero boundaries
    # Gaussian centered at T_n = 0 (20.5 C)
    sigma_T = 0.5
    temp_gauss = np.exp(-(T_n ** 2) / (2 * sigma_T ** 2))
    
    # Hard zero boundaries:
    # Below T_n = -0.8 (approx 4.7 C) or above T_n = 0.8 (approx 37.1 C), growth is zero.
    # This creates a "plateau" with hard edges.
    T_min = -0.8
    T_max = 0.8
    temp_mask = np.clip((T_n - T_min) * (T_max - T_n), 0.0, 1.0)
    
    # Combine: smooth shape inside, hard zero outside
    temp_factor = temp_gauss * temp_mask
    
    # --- pH Response ---
    # Combine smooth Gaussian with hard zero boundaries
    # Gaussian centered at pH_n = 0 (7.0 pH)
    sigma_pH = 0.5
    pH_gauss = np.exp(-(pH_n ** 2) / (2 * sigma_pH ** 2))
    
    # Hard zero boundaries:
    # Below pH_n = -0.6 (approx 4.0 pH) or above pH_n = 0.6 (approx 10.0 pH), growth is zero.
    pH_min = -0.6
    pH_max = 0.6
    pH_mask = np.clip((pH_n - pH_min) * (pH_max - pH_n), 0.0, 1.0)
    
    # Combine: smooth shape inside, hard zero outside
    pH_factor = pH_gauss * pH_mask
    
    # --- Biomass Inhibition ---
    # Inverse quadratic or similar
    inhib_b = 1.0 / (1.0 + p5 * b_n ** 2)
    
    # Construct Basis Functions
    # We have 10 parameters. We need 10 basis functions.
    
    # Basis 0: Bias
    basis0 = np.ones_like(b_n)
    
    # Basis 1: Substrate effect (Monod)
    basis1 = monod_s
    
    # Basis 2: Temperature effect (smooth + hard)
    basis2 = temp_factor
    
    # Basis 3: pH effect (smooth + hard)
    basis3 = pH_factor
    
    # Basis 4: Substrate * Temperature
    basis4 = monod_s * temp_factor
    
    # Basis 5: Substrate * pH
    basis5 = monod_s * pH_factor
    
    # Basis 6: Temperature * pH
    basis6 = temp_factor * pH_factor
    
    # Basis 7: Main Interaction (Substrate * Temp * pH)
    basis7 = monod_s * temp_factor * pH_factor
    
    # Basis 8: Biomass inhibition on main term
    basis8 = basis7 * inhib_b
    
    # Basis 9: Linear biomass effect (to capture simple density dependence)
    basis9 = b_n * temp_factor * pH_factor
    
    # Combine linearly
    prediction = (
        p0 * basis0 +
        p1 * basis1 +
        p2 * basis2 +
        p3 * basis3 +
        p4 * basis4 +
        p5 * basis5 +
        p6 * basis6 +
        p7 * basis7 +
        p8 * basis8 +
        p9 * basis9
    )
    
    # Ensure prediction is finite
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
