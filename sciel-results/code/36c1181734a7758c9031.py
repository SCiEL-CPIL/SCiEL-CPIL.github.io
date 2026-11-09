import numpy as np
import time

# Normalization constants based on training range
# x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physically-motivated nonlinear oscillator model using a potential energy gradient
    and velocity-dependent damping structure, expanded in a Chebyshev-like basis.
    
    The model form is: a = -dV/dx - gamma(v) * v
    Where V(x) is a polynomial potential and gamma(v) is a smooth, bounded damping function.
    
    To allow the optimizer flexibility while maintaining structure, we expand:
    - The restoring force term -dV/dx as a linear combination of odd Chebyshev polynomials of x.
    - The damping term -gamma(v) * v as a linear combination of odd Chebyshev polynomials of v.
    - Cross terms to capture any coupling not strictly separable.
    
    Inputs:
    x: position
    v: velocity
    params: 10 coefficients
    
    Output:
    Prediction of the acceleration.
    """
    # Normalize inputs to [-1, 1] to improve conditioning
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Clip to ensure stability in Chebyshev evaluation if slightly out of bounds
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Precompute Chebyshev polynomials for x and v
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    
    # For x (Restoring Force part)
    T0_x = np.ones_like(xn)
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    T4_x = 8.0 * xn**4 - 8.0 * xn**2 + 1.0
    
    # For v (Damping part)
    T0_v = np.ones_like(vn)
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn
    T4_v = 8.0 * vn**4 - 8.0 * vn**2 + 1.0
    
    # Construct 10 basis terms
    # We allocate 4 terms for the x-dependent restoring force (odd symmetry preferred, but allow even for asymmetry)
    # We allocate 4 terms for the v-dependent damping (odd symmetry preferred)
    # We allocate 2 cross terms for coupling
    
    # Restoring force terms (functions of x only)
    # Using T1, T3, T2, T4 to span the space. 
    # Note: A pure potential V(x) would imply dV/dx is a function of x.
    # T1 and T3 are odd, T2 and T4 are even. Including both allows asymmetric potentials.
    b0 = T1_x
    b1 = T3_x
    b2 = T2_x
    b3 = T4_x
    
    # Damping terms (functions of v only)
    # Damping is typically proportional to v, so odd functions of v are natural.
    # T1, T3 are odd. T2, T4 are even. 
    # The term is gamma(v)*v. If gamma is even, the term is odd. 
    # Let's use T1, T3, T2, T4 of v to represent the full damping profile -gamma(v)*v.
    b4 = T1_v
    b5 = T3_v
    b6 = T2_v
    b7 = T4_v
    
    # Cross terms (Coupling)
    # Allow interaction between position and velocity
    b8 = T1_x * T1_v
    b9 = T2_x * T1_v
    
    # Combine with parameters
    result = (params[0] * b0 + 
              params[1] * b1 + 
              params[2] * b2 + 
              params[3] * b3 + 
              params[4] * b4 + 
              params[5] * b5 + 
              params[6] * b6 + 
              params[7] * b7 + 
              params[8] * b8 + 
              params[9] * b9)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
