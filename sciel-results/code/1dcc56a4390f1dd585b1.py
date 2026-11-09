"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Piecewise-linear zigzag basis for modeling yield points and work-hardening segments.
# Uses smooth abs-based kink functions that BFGS can optimize.
#
# Model structure:
#   stress = base_linear + kink1 + kink2 + kink3 + temp_correction
#
# Where kink_i = a_i * max(0, strain - b_i) * (1 - c_i * max(0, strain - b_i))
# This creates piecewise-linear segments with different slopes after each breakpoint b_i.
# The (1 - c_i * ...) factor allows the slope to change after each yield point.
#
# Parameters:
#   p0: intercept
#   p1: initial elastic slope
#   p2: first yield strain
#   p3: slope change after first yield (hardening rate)
#   p4: second yield strain  
#   p5: slope change after second yield
#   p6: third yield strain
#   p7: slope change after third yield
#   p8: temperature coefficient (linear softening)
#   p9: temperature-strain interaction

def _smooth_kink(x, breakpoint, amplitude, slope_decay):
    """Smooth piecewise-linear kink: amplitude * max(0, x - breakpoint) * (1 - slope_decay * max(0, x - breakpoint))
    
    For x < breakpoint: 0
    For x >= breakpoint: amplitude * (x - breakpoint) - amplitude * slope_decay * (x - breakpoint)^2
    This is a parabolic segment that starts with slope=amplitude at the breakpoint
    and the slope decreases linearly, creating a "kink" in the derivative.
    """
    d = np.maximum(0.0, x - breakpoint)
    return amplitude * d * (1.0 - slope_decay * d)

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear zigzag basis with 3 kink points to model
    elastic region, yield points, and work-hardening segments.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base linear elastic term
    stress = p0 + p1 * strain
    
    # Kink 1: first yield point at strain=p2, with hardening characterized by p3
    # p3 controls the curvature/slope change after yield
    stress += _smooth_kink(strain, p2, p3, 2.0 * p3 / (1.0 + 0.1 * np.abs(p3)))
    
    # Kink 2: second yield/transition at strain=p4
    stress += _smooth_kink(strain, p4, p5, 2.0 * p5 / (1.0 + 0.1 * np.abs(p5)))
    
    # Kink 3: third transition at strain=p6
    stress += _smooth_kink(strain, p6, p7, 2.0 * p7 / (1.0 + 0.1 * np.abs(p7)))
    
    # Temperature effects: linear softening + interaction with strain
    stress += p8 * temp * (1.0 + p9 * strain)
    
    return stress
# EVOLVE-BLOCK-END
