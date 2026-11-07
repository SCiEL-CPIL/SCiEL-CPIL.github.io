"""Symbolic regression: E. coli growth rate with piecewise-linear plateau and rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structure:
    - Substrate: Monod-like saturation s / (Ks + s)
    - Temperature: piecewise-linear with hard zero below Tmin and above Tmax,
      linear rise to optimum, then linear decline to zero.
    - pH: piecewise-linear with hard zero below pHmin and above pHmax,
      linear rise to optimum, then linear decline to zero.
    - Density: rational inhibition b / (Kb + b) to model crowding effects.
    - Overall: product of these factors scaled by a max growth rate.
    
    Parameters (10 total):
    p0: max growth rate scale
    p1: substrate half-saturation Ks
    p2: temperature minimum (below this, growth = 0)
    p3: temperature optimum
    p4: temperature maximum (above this, growth = 0)
    p5: pH minimum (below this, growth = 0)
    p6: pH optimum
    p7: pH maximum (above this, growth = 0)
    p8: density half-inhibition Kb
    p9: offset/bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure positive denominators to avoid division by zero
    Ks = np.maximum(p1, 0.01)
    Kb = np.maximum(p8, 0.01)

    # Substrate saturation (Monod)
    substrate_factor = s / (Ks + s)

    # Temperature response: piecewise linear
    # Below Tmin: 0
    # Between Tmin and T_opt: linear rise from 0 to 1
    # Between T_opt and Tmax: linear decline from 1 to 0
    # Above Tmax: 0
    # Use safe ranges to avoid division by zero
    T_min = np.minimum(p2, p3 - 0.1)
    T_max = np.maximum(p4, p3 + 0.1)
    
    T_span_low = np.maximum(T_opt - T_min, 0.1) if (T_opt := p3) else 0.1
    T_span_high = np.maximum(T_max - T_opt, 0.1)
    
    # Compute temperature factor safely
    T_factor = np.zeros_like(temp, dtype=float)
    
    # Rising phase: T_min <= temp <= T_opt
    mask_rise = (temp >= T_min) & (temp <= T_opt)
    T_factor[mask_rise] = (temp[mask_rise] - T_min) / T_span_low
    
    # Falling phase: T_opt < temp <= T_max
    mask_fall = (temp > T_opt) & (temp <= T_max)
    T_factor[mask_fall] = (T_max - temp[mask_fall]) / T_span_high
    
    # Clamp to [0, 1]
    T_factor = np.clip(T_factor, 0.0, 1.0)

    # pH response: piecewise linear
    pH_min = np.minimum(p5, p6 - 0.1)
    pH_max = np.maximum(p7, p6 + 0.1)
    
    pH_span_low = np.maximum(p6 - pH_min, 0.1)
    pH_span_high = np.maximum(pH_max - p6, 0.1)
    
    pH_factor = np.zeros_like(pH, dtype=float)
    
    mask_pH_rise = (pH >= pH_min) & (pH <= p6)
    pH_factor[mask_pH_rise] = (pH[mask_pH_rise] - pH_min) / pH_span_low
    
    mask_pH_fall = (pH > p6) & (pH <= pH_max)
    pH_factor[mask_pH_fall] = (pH_max - pH[mask_pH_fall]) / pH_span_high
    
    pH_factor = np.clip(pH_factor, 0.0, 1.0)

    # Density inhibition: b / (Kb + b) approaches 1 as b grows, but we want inhibition
    # Actually, for inhibition, we might want 1 / (1 + b/Kb) or similar
    # Let's use: density_factor = Kb / (Kb + b) which is 1 at b=0 and decreases
    density_factor = Kb / (Kb + b)

    # Combine all factors
    growth = p0 * substrate_factor * T_factor * pH_factor * density_factor + p9

    # Ensure output is finite
    growth = np.where(np.isfinite(growth), growth, 0.0)

    return growth
# EVOLVE-BLOCK-END
