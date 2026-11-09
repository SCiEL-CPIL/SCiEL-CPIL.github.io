"""
Fourier-Chebyshev Orthogonal Basis Oscillator Model:
Replaces the previous sparse tanh/sin basis with a systematic, orthogonal basis:
1. Chebyshev polynomials of the first kind T_k(x) for position nonlinearity.
2. Chebyshev polynomials of the first kind T_k(v) for velocity nonlinearity.
3. Fourier series sin(k*t), cos(k*t) for periodic driving forces.
4. Cross terms T_i(x)*T_j(v) to capture nonlinear position-velocity couplings.

This structure provides:
- Better conditioning for BFGS due to orthogonality (near-identity Gram matrix).
- Higher expressivity for nonlinear dynamics within the same 10-parameter budget.
- Explicit separation of spatial (x, v) and temporal (t) dynamics.
- Robust convergence from the all-ones initialization.
"""
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation:
    y = p0 + p1*T1(x) + p2*T2(x) + p3*T3(x) 
      + p4*T1(v) + p5*T2(v) + p6*T3(v)
      + p7*sin(t) + p8*cos(t)
      + p9*T1(x)*T1(v)
    
    Where T_k are Chebyshev polynomials:
    T_0(x) = 1
    T_1(x) = x
    T_2(x) = 2x^2 - 1
    T_3(x) = 4x^3 - 3x
    
    Note: T_0 is absorbed into p0.
    The cross term T1(x)*T1(v) = x*v captures linear coupling.
    Higher order cross terms are omitted to save parameters for higher-order 
    univariate nonlinearities which are often more significant in oscillators.
    
    Input ranges:
    t: [30, 50]
    x: [-0.2056, 0.1749]
    v: [-0.3391, 0.3624]
    
    The small ranges of x and v ensure Chebyshev polynomials are well-behaved 
    and do not suffer from numerical instability, but we normalize them to [-1, 1] 
    internally to maximize the effectiveness of the Chebyshev basis, which is 
    defined on [-1, 1].
    """
    
    # Normalize x and v to [-1, 1] for Chebyshev stability
    # x range approx [-0.21, 0.17], center -0.02, width 0.38
    # v range approx [-0.34, 0.36], center 0.01, width 0.70
    # Using fixed normalization based on training ranges to avoid data-dependent params
    
    x_norm = (x - (-0.015)) / 0.20  # Roughly maps [-0.21, 0.18] to [-1, 1]
    v_norm = (v - 0.01) / 0.35      # Roughly maps [-0.34, 0.36] to [-1, 1]
    
    # Clip to prevent overflow in high-order polys if out of bounds (though inputs are in range)
    x_norm = np.clip(x_norm, -1.1, 1.1)
    v_norm = np.clip(v_norm, -1.1, 1.1)
    
    # Chebyshev polynomials of the first kind
    T1_x = x_norm
    T2_x = 2 * x_norm**2 - 1
    T3_x = 4 * x_norm**3 - 3 * x_norm
    
    T1_v = v_norm
    T2_v = 2 * v_norm**2 - 1
    T3_v = 4 * v_norm**3 - 3 * v_norm
    
    # Cross term: Linear coupling in normalized space
    T1_xv = T1_x * T1_v
    
    # Fourier terms for time
    # t is in [30, 50]. The fundamental frequency of the oscillator is unknown.
    # Using low frequency components.
    # If the oscillator has a period T, frequency w = 2*pi/T.
    # We use sin(t) and cos(t) as a generic periodic driver. 
    # To be more robust, we might scale t. 
    # However, without knowing the period, standard sin(t) is a reasonable first guess.
    # Let's use a scaled time to allow BFGS to effectively find the frequency 
    # via the coefficients if we had more params, but here we fix the frequency.
    # Given the task is a "nonlinear oscillator", the driving force might be 
    # periodic. Let's assume the main dynamics are captured by the x/v terms 
    # and the time terms capture any external forcing or phase offset.
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    pred = (p0 
            + p1 * T1_x 
            + p2 * T2_x 
            + p3 * T3_x 
            + p4 * T1_v 
            + p5 * T2_v 
            + p6 * T3_v 
            + p7 * sin_t 
            + p8 * cos_t 
            + p9 * T1_xv)
            
    return pred

# EVOLVE-BLOCK-END
