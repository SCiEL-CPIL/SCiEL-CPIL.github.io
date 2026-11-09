"""Piecewise-linear sawtooth basis for aluminium stress-strain with temperature."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear basis built from abs to model yield points and
    work-hardening segments. Temperature modulates the elastic modulus and
    shifts the yield breakpoint.
    
    params layout (10 constants):
      p0: elastic modulus scale
      p1: temperature effect on modulus
      p2: intercept
      p3: yield strain breakpoint (base)
      p4: slope change at yield (plastic hardening)
      p5: second breakpoint
      p6: slope change at second breakpoint
      p7: third breakpoint
      p8: slope change at third breakpoint
      p9: temperature effect on yield breakpoint
    """
    s = strain
    t = temp
    p = params
    
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    p6 = p[6]
    p7 = p[7]
    p8 = p[8]
    p9 = p[9]
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * t
    
    # Temperature-shifted yield breakpoint
    yield_bp = p3 + p9 * t
    
    # ReLU-like functions: max(0, x) = (x + |x|) / 2
    r1 = (s - yield_bp + np.abs(s - yield_bp)) * 0.5
    r2 = (s - p5 + np.abs(s - p5)) * 0.5
    r3 = (s - p7 + np.abs(s - p7)) * 0.5
    
    # Elastic region: linear with temperature-dependent modulus
    elastic = E * s
    
    # Plastic hardening segments
    plastic = p4 * r1 + p6 * r2 + p8 * r3
    
    # Base offset
    offset = p2
    
    stress = elastic + plastic + offset
    
    return stress
# EVOLVE-BLOCK-END
