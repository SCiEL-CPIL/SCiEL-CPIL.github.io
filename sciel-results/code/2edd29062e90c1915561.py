"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear "sawtooth" basis to model discrete yield points
    and work-hardening segments, combined with temperature-dependent elastic
    and plastic terms.
    
    Model:
    sigma = E(T)*eps + Y(T)*[sawtooth(eps, p5, p6) + p7*eps + p8*eps^2] + p9
    E(T) = p0 + p1*T + p2*T^2
    Y(T) = p3 + p4*T
    sawtooth(x, a, b) = a * (x - floor(x/b) * b) - a*b/2
    
    params (10 constants):
      p0, p1, p2: Elastic modulus coefficients
      p3, p4: Yield/Hardening scale coefficients
      p5: Sawtooth amplitude (hardening magnitude per segment)
      p6: Sawtooth period (strain per hardening segment)
      p7: Linear plastic strain correction
      p8: Quadratic strain correction
      p9: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    Y_T = p3 + p4 * T
    
    # Piecewise-linear sawtooth basis: creates kinked transitions
    # at regular strain intervals, modeling discrete yield/hardening segments
    # Use abs-based construction: sawtooth(x) = x - b*floor(x/b) - b/2
    # To keep it differentiable-ish for BFGS, use a smooth approximation
    # via min/max operations that BFGS can handle
    
    # Safe period: avoid division by zero
    period = np.where(np.abs(p6) > 1e-8, p6, 1.0)
    
    # Sawtooth: triangular wave with kinks at multiples of period
    # x_mod = x - period * floor(x / period)
    # For numerical stability and differentiability, use:
    # tri(x) = period * (2 * abs(x/period - 0.5) - 1) ... but this has kinks
    # Use a min-based construction:
    # saw = p5 * (eps - period * np.floor(eps / period))
    # But floor is not differentiable. Use a smooth approximation via tanh-based
    # step, or just use abs which BFGS can handle with subgradients.
    
    # Simplest robust piecewise-linear: use abs to create V-shapes
    # Combined with linear terms to create sawtooth-like behavior
    # seg = eps - period * np.round(eps / period)  # triangle wave, has kinks
    # But round/floor cause issues for gradient-based optimization.
    
    # Alternative: use min/max to create piecewise linear functions
    # h1 = np.maximum(eps, 0) - np.maximum(eps - period, 0)  # ramp up then flat
    # This is differentiable almost everywhere, BFGS can handle it
    
    # Create a "hardening segment" function:
    # Increases linearly with slope 1 up to period, then stays constant
    # This models elastic-plastic transition with saturation
    seg1 = np.maximum(eps, 0) - np.maximum(eps - period, 0)
    # Normalize to [0, 1] range
    seg1 = seg1 / np.where(np.abs(period) > 1e-8, np.abs(period), 1.0)
    
    # Second segment: starts after first, models continued hardening
    seg2 = np.maximum(eps - period, 0) - np.maximum(eps - 2*period, 0)
    seg2 = seg2 / np.where(np.abs(period) > 1e-8, np.abs(period), 1.0)
    
    # Combine segments with decreasing amplitude (work hardening saturation)
    plastic_shape = p5 * (seg1 + 0.5 * seg2)
    
    # Add linear and quadratic corrections for overall trend
    linear_plastic = p7 * eps
    quad_plastic = p8 * eps * eps
    
    # Core model
    elastic_term = E_T * eps
    plastic_term = Y_T * (plastic_shape + linear_plastic + quad_plastic)
    
    sigma = elastic_term + plastic_term + p9
    
    return sigma
# EVOLVE-BLOCK-END
