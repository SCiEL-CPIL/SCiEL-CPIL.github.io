"""
REFINED PRODUCT BASIS: Chebyshev-polynomial product basis with expanded
nonlinear activations for richer function space coverage.

Key improvements over previous version:
- Replace sigmoid (low information on [-1,1]) with x*cos(x) for oscillatory even
- Replace T2*tanh with x*exp(-x^2) for asymmetric localized feature
- Add x*sin(2x) for higher-frequency oscillatory component
- Keep core polynomial + tanh + sin + Gaussian structure
- All terms linear in params => convex BFGS problem

Basis functions (10 total), where x' = x/3 maps [-3,3] to [-1,1]:
- T0(x')                  (constant)
- T1(x')                  (linear)
- T2(x')                  (quadratic)
- T3(x')                  (cubic)
- T4(x')                  (quartic)
- T1(x') * tanh(x')       (smooth odd nonlinearity)
- T1(x') * sin(x')        (oscillatory odd)
- T1(x') * cos(x')        (oscillatory even)
- exp(-x'^2)              (Gaussian bump - local flexibility)
- T1(x') * exp(-x'^2)     (asymmetric bump - captures skew)
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    xp = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(xp)
    T1 = xp
    T2 = 2.0 * xp * xp - 1.0
    T3 = 4.0 * xp**3 - 3.0 * xp
    T4 = 8.0 * xp**4 - 8.0 * xp**2 + 1.0
    
    # Nonlinear activations
    tanh_x = np.tanh(xp)
    sin_x = np.sin(xp)
    cos_x = np.cos(xp)
    # Gaussian bump: exp(-x'^2), ranges from exp(-1)~0.37 to 1.0
    gauss = np.exp(-xp * xp)
    
    # Product basis functions
    b0 = T0 * 1.0              # constant
    b1 = T1 * 1.0              # linear
    b2 = T2 * 1.0              # quadratic
    b3 = T3 * 1.0              # cubic
    b4 = T4 * 1.0              # quartic
    b5 = T1 * tanh_x           # smooth odd nonlinearity
    b6 = T1 * sin_x            # oscillatory odd
    b7 = T1 * cos_x            # oscillatory even
    b8 = gauss                 # Gaussian bump (even, local)
    b9 = T1 * gauss            # asymmetric bump (odd, local)
    
    # Linear combination
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
