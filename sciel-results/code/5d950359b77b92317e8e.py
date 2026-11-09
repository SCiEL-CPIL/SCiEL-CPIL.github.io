"""Piecewise-linear ramp model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: a sum of clipped linear (ramp) functions to capture yield point
    and hardening, plus a linear elastic term.
    
    stress = p0*strain + sum_i p_i * max(0, strain - t_i)
    where t_i = p_{5+i} for i in 0..3 (yield strain thresholds)
    and p_1..p_4 are the slopes of the ramp functions.
    
    Temperature dependence is folded into the ramp thresholds and slopes
    via linear corrections.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic term
    elastic = p0 * strain
    
    # Temperature-adjusted thresholds for the ramps
    t1 = p5 + p8 * temp
    t2 = p6 + p9 * temp
    t3 = p7
    t4 = p8
    
    # Ramp functions: max(0, strain - threshold)
    ramp1 = np.maximum(0.0, strain - t1)
    ramp2 = np.maximum(0.0, strain - t2)
    ramp3 = np.maximum(0.0, strain - t3)
    ramp4 = np.maximum(0.0, strain - t4)
    
    # Weighted sum of ramps
    plastic = p1 * ramp1 + p2 * ramp2 + p3 * ramp3 + p4 * ramp4
    
    prediction = elastic + plastic
    
    return prediction
# EVOLVE-BLOCK-END
