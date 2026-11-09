"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# The evaluator calls equation(t, x, v, params).
# We use a Chebyshev polynomial basis in normalized state variables (x, v)
# to approximate the acceleration field of a nonlinear oscillator.
# This leverages superior conditioning and global approximation properties
# for bounded nonlinear dynamics compared to monomial polynomials.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Chebyshev polynomial basis in (x, v).
    
    We normalize x and v to [-1, 1] based on training ranges,
    then use a total-degree Chebyshev expansion with cross terms.
    
    Training ranges:
      x: [-0.2056, 0.1749] -> center = -0.01535, half_range = 0.19025
      v: [-0.3391, 0.3624] -> center = 0.01165, half_range = 0.35075
    
    We use 10 parameters for a structured Chebyshev basis:
    T0(x)T0(v), T1(x)T0(v), T0(x)T1(v), T2(x)T0(v), T0(x)T2(v),
    T1(x)T1(v), T3(x)T0(v), T0(x)T3(v), T2(x)T1(v), T1(x)T2(v)
    
    Additionally, we include time-dependent forcing via sin(t) and cos(t)
    multiplied by the leading Chebyshev terms to capture periodic driving.
    
    Actually, let's use a cleaner decomposition:
    - 4 params for time-dependent forcing: a0*sin(t) + a1*cos(t) + a2*sin(t)*x_norm + a3*cos(t)*v_norm
    - 6 params for the static acceleration field via Chebyshev in (x_norm, v_norm)
    
    Total: 10 parameters.
    """
    # Normalize x and v to approximately [-1, 1]
    # x: center = (−0.2056 + 0.1749)/2 = -0.01535, half_range = (0.1749 - (-0.2056))/2 = 0.19025
    # v: center = (−0.3391 + 0.3624)/2 = 0.01165, half_range = (0.3624 - (-0.3391))/2 = 0.35075
    
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    # Avoid division by zero if ranges are degenerate
    x_norm = (x - x_center) / x_half
    v_norm = (v - v_center) / v_half
    
    # Clip to [-1, 1] for stability (Chebyshev polynomials grow outside [-1,1])
    x_norm = np.clip(x_norm, -1.0, 1.0)
    v_norm = np.clip(v_norm, -1.0, 1.0)
    
    # Compute Chebyshev polynomials up to degree 3
    # T0(z) = 1
    # T1(z) = z
    # T2(z) = 2z^2 - 1
    # T3(z) = 4z^3 - 3z
    
    T0_x = np.ones_like(x_norm)
    T1_x = x_norm
    T2_x = 2.0 * x_norm * x_norm - 1.0
    T3_x = 4.0 * x_norm * x_norm * x_norm - 3.0 * x_norm
    
    T0_v = np.ones_like(v_norm)
    T1_v = v_norm
    T2_v = 2.0 * v_norm * v_norm - 1.0
    T3_v = 4.0 * v_norm * v_norm * v_norm - 3.0 * v_norm
    
    # Time-dependent forcing terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Build the prediction:
    # Params 0-3: Time-dependent forcing
    # Params 4-9: Static Chebyshev acceleration field
    
    # Time-dependent part:
    # p0 * sin(t) + p1 * cos(t) + p2 * sin(t) * T1_x + p3 * cos(t) * T1_v
    
    # Static part (Chebyshev basis in x, v):
    # p4 * T0_x * T0_v  (constant)
    # p5 * T1_x * T0_v  (linear in x)
    # p6 * T0_x * T1_v  (linear in v)
    # p7 * T2_x * T0_v  (quadratic in x)
    # p8 * T0_x * T2_v  (quadratic in v)
    # p9 * T1_x * T1_v  (cross term)
    
    y = (params[0] * sin_t + params[1] * cos_t +
         params[2] * sin_t * T1_x + params[3] * cos_t * T1_v +
         params[4] * T0_x * T0_v +
         params[5] * T1_x * T0_v +
         params[6] * T0_x * T1_v +
         params[7] * T2_x * T0_v +
         params[8] * T0_x * T2_v +
         params[9] * T1_x * T1_v)
    
    return y
# EVOLVE-BLOCK-END
