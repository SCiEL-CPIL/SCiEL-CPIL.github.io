"""E. coli growth rate model with smooth, linear-in-parameters biological basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a linear-in-parameters combination of
    smooth biological basis functions.
    
    All basis functions are smooth (differentiable) to support BFGS optimization.
    The model is linear in parameters, ensuring reliable convergence.
    
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
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    
    # Smooth temperature activation using sigmoid-based bell curve
    # Rises from ~0 below -0.5, peaks near 0.1, declines above 0.3
    T_rise = 1.0 / (1.0 + np.exp(-12.0 * (T_n + 0.4)))
    T_fall = 1.0 / (1.0 + np.exp(-12.0 * (0.3 - T_n)))
    T_factor = T_rise * T_fall
    
    # Smooth pH activation: bell-shaped around pH 7
    pH_factor = 1.0 / (1.0 + np.exp(8.0 * (pH_n - 0.0)) * np.exp(-8.0 * (pH_n + 0.6)))
    # Alternative: use a smooth Gaussian-like shape
    pH_factor = np.exp(-4.0 * pH_n ** 2)
    
    # Substrate saturation: Monod-like (smooth)
    monod_s = s / (2.0 + s)
    
    # Density inhibition: smooth rational
    b_inhib = 1.0 / (1.0 + 0.5 * b_n ** 2)
    
    # Additional smooth basis functions
    # Temperature linear
    T_lin = T_n
    # Temperature squared
    T_quad = T_n ** 2
    # pH linear
    pH_lin = pH_n
    # Substrate normalized
    s_lin = s_n
    # Biomass linear
    b_lin = b_n
    # Interaction: temp * pH
    T_pH_int = T_n * pH_n
    # Interaction: substrate * temp
    s_T_int = monod_s * T_n
    # Interaction: substrate * pH
    s_pH_int = monod_s * pH_n
    # Triple interaction
    main_term = monod_s * T_factor * pH_factor * b_inhib
    
    # Construct full prediction (linear in params)
    prediction = (
        p0 +
        p1 * main_term +
        p2 * T_factor +
        p3 * pH_factor +
        p4 * monod_s +
        p5 * b_inhib +
        p6 * T_lin +
        p7 * s_lin +
        p8 * pH_lin +
        p9 * b_lin
    )
    
    # Ensure prediction is finite
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
