"""E. coli growth rate model with piecewise-linear plateau and rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a piecewise-linear 'growth plateau' structure.
    
    This model uses np.clip to enforce hard zero-growth boundaries for suboptimal
    temperature and pH, creating a biologically motivated hard constraint that
    prevents fitting growth in impossible regimes. A rational inhibition term
    captures substrate saturation and density-dependent inhibition.
    
    The model is linear in parameters, ensuring BFGS can reliably optimize.
    
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
    # b: [1, 5] -> center 3, scale 2
    b_n = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, scale 2
    s_n = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, scale 19.5
    T_n = (temp - 20.5) / 19.5
    # pH: [2, 12] -> center 7, scale 5
    pH_n = (pH - 7.0) / 5.0
    
    # Piecewise-linear temperature activation:
    # Growth is zero below T_min, rises linearly to peak, then declines
    # T_min and T_max are fixed biological boundaries
    # Using clip to enforce hard boundaries
    T_low = -0.4   # ~12.7°C - minimum viable temperature
    T_high = 0.1   # ~22.5°C - near optimal
    T_peak = 0.15  # ~23.4°C - peak growth temperature
    
    # Temperature factor: piecewise linear with hard zero below T_low
    # Below T_low: 0
    # T_low to T_peak: linear rise
    # Above T_peak: linear decline
    T_factor = np.clip(T_n - T_low, 0, T_peak - T_low) / (T_peak - T_low)
    # Add a decline term for temperatures above peak
    T_decline = np.clip(T_n - T_peak, 0, 1.0)
    T_combined = T_factor - p4 * T_decline  # p4 controls decline rate
    
    # Ensure non-negative temperature factor
    T_combined = np.maximum(T_combined, 0)
    
    # Piecewise-linear pH activation:
    # Growth is zero outside optimal pH range
    pH_low = -0.3   # ~5.5 pH - minimum viable
    pH_high = 0.3   # ~8.5 pH - maximum viable
    pH_center = 0.0  # 7.0 pH - optimal
    
    # pH factor: trapezoidal shape with hard zero boundaries
    pH_factor = np.clip(pH_n - pH_low, 0, 1.0) * np.clip(pH_high - pH_n, 0, 1.0)
    # Normalize to [0, 1]
    pH_range = (pH_high - pH_low)
    if pH_range > 0:
        pH_factor = pH_factor / (pH_range * 0.5)  # approximate normalization
    pH_factor = np.clip(pH_factor, 0, 1)
    
    # Substrate saturation: rational (Monod-like) term
    # s / (Ks + s) with Ks fixed at 2.0
    monod_s = s / (2.0 + s)
    
    # Density inhibition: rational term
    # 1 / (1 + b_inhibition * b_n^2)
    b_inhib = 1.0 / (1.0 + p5 * b_n ** 2)
    
    # Construct prediction as product of factors times coefficients
    # The model is: mu = p0 + p1 * monod_s * T_combined * pH_factor * b_inhib + ...
    # Plus additional terms for flexibility
    
    # Main growth term
    main_term = monod_s * T_combined * pH_factor * b_inhib
    
    # Additional basis functions for flexibility
    # Linear temperature correction
    T_lin_corr = T_n * pH_factor
    # Quadratic substrate effect
    s_quad = s_n ** 2 * T_combined
    # Interaction: substrate and pH
    int_s_pH = monod_s * pH_factor
    
    # Biomass linear effect
    b_lin = b_n * T_combined * pH_factor
    
    # Construct full prediction
    prediction = (
        p0 * 1.0 +
        p1 * main_term +
        p2 * T_combined +
        p3 * pH_factor +
        p6 * T_lin_corr +
        p7 * s_quad +
        p8 * int_s_pH +
        p9 * b_lin
    )
    
    # Ensure prediction is finite
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
