"""Symbolic regression for a nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model-inspired kernel basis for nonlinear oscillator acceleration.
    
    Two Gaussian bumps in (x, v) phase space with shared width, plus a flexible
    periodic time-dependent forcing term with fitted amplitude, frequency, and phase.
    
    Parameters:
    p0: weight of bump 1
    p1: center x of bump 1 (scaled via tanh)
    p2: center v of bump 1 (scaled via tanh)
    p3: weight of bump 2
    p4: center x of bump 2 (scaled via tanh)
    p5: center v of bump 2 (scaled via tanh)
    p6: log shared-width (scaled via exp)
    p7: amplitude of periodic time term
    p8: frequency of periodic time term (scaled)
    p9: phase of periodic time term
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1] for better conditioning
    ts = (t - 40.0) / 10.0
    xs = x / 0.2
    vs = v / 0.35
    
    # Transform parameters to valid ranges
    # Centers: map to [-2, 2] to cover the scaled input range with margin
    cx1 = 2.0 * np.tanh(p1)
    cv1 = 2.0 * np.tanh(p2)
    cx2 = 2.0 * np.tanh(p4)
    cv2 = 2.0 * np.tanh(p5)
    
    # Shared width: ensure positive and not too small
    sigma = np.exp(np.clip(p6, -5, 5))
    
    # Calculate Gaussian bumps with shared width
    dx1 = xs - cx1
    dv1 = vs - cv1
    dist1_sq = dx1**2 + dv1**2
    G1 = np.exp(-0.5 * dist1_sq / (sigma**2))
    
    dx2 = xs - cx2
    dv2 = vs - cv2
    dist2_sq = dx2**2 + dv2**2
    G2 = np.exp(-0.5 * dist2_sq / (sigma**2))
    
    # Flexible time-dependent term: amplitude * sin(freq * ts + phase)
    # Frequency scaled: p8 starts at 1, so initial freq ~ 1 rad per scaled time unit
    # Phase: p9 starts at 1 rad
    freq = 5.0 * np.tanh(p8) + 1.0  # ensures positive, ranges roughly [1-5, 1+5]
    phase = p9
    time_term = p7 * np.sin(freq * ts + phase)
    
    # Combine
    a = p0 * G1 + p3 * G2 + time_term
    
    # Ensure finite output
    if not np.all(np.isfinite(a)):
        return np.zeros_like(a)
        
    return a
# EVOLVE-BLOCK-END
