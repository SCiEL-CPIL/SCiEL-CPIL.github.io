"""
COMBINED APPROACH: Chebyshev-polynomial product basis with diverse nonlinear
activations (tanh, sin, exp, sigmoid) to capture a wide range of smooth
functional forms.

Structural idea from current program:
- Use Chebyshev polynomials T_k(x') as stable global basis on [-1,1]
- Multiply by nonlinear activations to create product features
- Keep the model LINEAR in params so BFGS solves a convex least-squares problem

Refinement from inspirations:
- Include both tanh and sin activations (current has both, but we expand)
- Add a Gaussian-like localized term via exp(-x'^2) for local flexibility
- Add a sigmoid for saturation behavior
- Scale basis functions to comparable magnitudes for numerical conditioning
- Keep polynomial terms up to degree 4 for global trend capture

Basis functions (10 total), where x' = x/3 maps [-3,3] to [-1,1]:
- T0(x') * 1              (constant)
- T1(x') * 1              (linear)
- T2(x') * 1              (quadratic)
- T3(x') * 1              (cubic)
- T4(x') * 1              (quartic)
- T1(x') * tanh(x')       (linear * tanh - odd nonlinearity)
- T2(x') * tanh(x')       (quadratic * tanh - even nonlinearity)
- T1(x') * sin(x')        (linear * sin - oscillatory)
- exp(-x'^2)              (Gaussian bump - local flexibility)
- 1/(1+exp(-2*x'))        (sigmoid - saturation)

All 10 parameters are linear in the output, so BFGS solves a convex problem.
The Chebyshev polynomials provide numerical stability, and the product with
diverse activations captures smooth nonlinear trends.
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
    # Gaussian bump: exp(-x'^2), ranges from exp(-1)~0.37 to 1.0
    gauss = np.exp(-xp * xp)
    # Sigmoid: 1/(1+exp(-2*x')), ranges from ~0.12 to ~0.88 on [-1,1]
    # Clip argument to prevent overflow
    sig_arg = np.clip(-2.0 * xp, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(sig_arg))
    
    # Product basis functions
    b0 = T0 * 1.0              # constant
    b1 = T1 * 1.0              # linear
    b2 = T2 * 1.0              # quadratic
    b3 = T3 * 1.0              # cubic
    b4 = T4 * 1.0              # quartic
    b5 = T1 * tanh_x           # linear * tanh
    b6 = T2 * tanh_x           # quadratic * tanh
    b7 = T1 * sin_x            # linear * sin
    b8 = gauss                 # Gaussian bump
    b9 = sigmoid               # sigmoid saturation
    
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
