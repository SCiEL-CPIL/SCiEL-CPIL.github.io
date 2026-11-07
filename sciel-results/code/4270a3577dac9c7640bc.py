"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear growth plateau model.
    
    This strategy uses hard constraints via np.clip to enforce zero growth outside
    optimal biological ranges, and rational terms for substrate and density inhibition.
    The model is linear in parameters, ensuring a convex optimization landscape for BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Substrate: Monod-like saturation s/(1+s)
    # p1 is the coefficient
    f_s = s / (1.0 + s)
    
    # 2. Temperature: Piecewise-linear plateau
    # Hard clip to 0 outside [30, 44] (approx E. coli growth range)
    # Linear ramp up from 30 to 37, plateau from 37 to 41, ramp down to 44
    # We use a combination of clips to create a trapezoid
    # Lower ramp: max(0, temp - 30)
    # Upper ramp: max(0, 44 - temp)
    # To make it a plateau, we can use: min(temp - 30, 44 - temp, 7)
    # But min is not differentiable everywhere. Let's use a smooth approximation or just linear terms.
    # Actually, np.clip is differentiable almost everywhere.
    # Let's define a trapezoid function using clips:
    # T_low = np.clip(temp - 30.0, 0.0, 7.0)  # Ramps up from 30 to 37, stays at 7
    # T_high = np.clip(44.0 - temp, 0.0, 7.0) # Ramps down from 44 to 37, stays at 7
    # Plateau = np.minimum(T_low, T_high)
    # This creates a trapezoid that is 0 below 30, ramps to 7 at 37, stays 7 until 41, ramps to 0 at 44.
    # This is piecewise linear.
    T_low = np.clip(temp - 30.0, 0.0, 7.0)
    T_high = np.clip(44.0 - temp, 0.0, 7.0)
    f_temp = np.minimum(T_low, T_high)
    
    # 3. pH: Piecewise-linear plateau
    # Hard clip to 0 outside [6.0, 8.0]
    # Linear ramp up from 6.0 to 7.0, plateau from 7.0 to 7.5, ramp down to 8.0
    P_low = np.clip(pH - 6.0, 0.0, 1.0)
    P_high = np.clip(8.0 - pH, 0.0, 1.0)
    f_ph = np.minimum(P_low, P_high)
    
    # 4. Density: Exponential inhibition exp(-0.5*b)
    # p4 is the amplitude
    # Fixed rate for stability
    f_b = p4 * np.exp(-0.5 * b)
    
    # 5. Interaction: Substrate * Temperature plateau
    # p5 is the coefficient
    f_st = p5 * f_s * f_temp
    
    # 6. Interaction: Substrate * pH plateau
    # p6 is the coefficient
    f_sp = p6 * f_s * f_ph
    
    # 7. Interaction: Temperature * pH plateau
    # p7 is the coefficient
    f_tp = p7 * f_temp * f_ph
    
    # 8. Linear term for substrate (to capture any non-saturation effects)
    # p8 is the coefficient
    f_s_lin = p8 * s
    
    # 9. Linear term for temperature (to capture any non-plateau effects)
    # p9 is the coefficient
    f_temp_lin = p9 * (temp - 37.0)
    
    # 10. Bias
    # p0
    
    # Combine additively
    # Note: p2 and p3 are used as coefficients for f_temp and f_ph respectively
    y = p0 + p1 * f_s + p2 * f_temp + p3 * f_ph + f_b + f_st + f_sp + f_tp + f_s_lin + f_temp_lin
    
    return y
# EVOLVE-BLOCK-END
