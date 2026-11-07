"""
COMBINED APPROACH: Adaptive product basis with learned frequency and width.

Structural idea from inspiration 1: use parameters to ADAPT the basis function
shapes (weights/frequencies/widths), not just linear coefficients. This lets
BFGS discover the correct oscillatory frequency, Gaussian width, etc., rather
than being stuck with a fixed basis that may not match the target.

Refinement machinery from current program + inspiration 2:
- Chebyshev polynomials on [-1,1] for numerical stability
- Product of Chebyshev with adaptive nonlinear activations
- Keep the model LINEAR in the final output combination where possible,
  but use some params to shape the basis

Design (10 params):
- p[0:4] = adaptive frequencies/scales for nonlinear terms
- p[4:10] = linear combination weights for the 6 basis functions

Basis functions (6 total, shapes depend on p[0:4]):
- b0 = 1                          (constant)
- b1 = T1(xp)                     (linear)
- b2 = T2(xp)                     (quadratic)
- b3 = T1(xp) * tanh(p[0]*xp)     (adaptive smooth odd)
- b4 = T1(xp) * sin(p[1]*xp)      (adaptive oscillatory odd)
- b5 = exp(-(p[2]*xp)^2)          (adaptive Gaussian bump)

Wait, that's only 6 basis functions but 10 params (4 shape + 6 weights).
Actually let me reconsider to use all 10 more effectively.

Better design: 5 basis functions, each with an adaptive parameter, plus 5 weights.
Or: use 10 params as 5 (weight, shape) pairs for 5 basis functions.

Let's do:
- b0 = T0 * w0 = 1 * p[0]  -- but that's just constant
Actually, let me think differently.

Use 10 params: 5 shape params and 5 weight params.
Basis:
- b0 = 1
- b1 = T1(xp)
- b2 = T2(xp)
- b3 = T1(xp) * tanh(p[5] * xp)
- b4 = exp(-(p[6] * xp)^2)

Weights: p[0:5]
Shapes: p[5:10] -- but we only need 2 shape params (for tanh freq and gauss width)

Let me use:
- p[0] = weight for constant
- p[1] = weight for linear
- p[2] = weight for quadratic
- p[3] = weight for adaptive tanh term
- p[4] = weight for adaptive Gaussian
- p[5] = frequency for tanh
- p[6] = width for Gaussian
- p[7] = weight for adaptive sin term
- p[8] = frequency for sin
- p[9] = weight for cubic

Basis:
- b0 = 1
- b1 = T1(xp)
- b2 = T2(xp)
- b3 = T3(xp)
- b4 = T1(xp) * tanh(p[5]*xp)
- b5 = T1(xp) * sin(p[8]*xp)
- b6 = exp(-(p[6]*xp)^2)

That's 7 basis functions but only 10 params total with 7 weights and 3 shapes.
Hmm, let me just go with a clean design.

FINAL DESIGN:
10 params split as:
- p[0:5] = weights for 5 basis functions
- p[5:10] = shape parameters (2 frequencies, 1 width, 1 tanh scale, 1 gaussian center)

Basis functions:
- b0 = 1
- b1 = T1(xp)
- b2 = T2(xp)
- b3 = T1(xp) * tanh(p[5] * xp)
- b4 = exp(-p[7] * (xp - p[9])^2)  -- adaptive Gaussian with center

This gives 5 weights + 5 shape params = 10 total.
"""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    xp = x / 3.0
    
    # Compute Chebyshev polynomials
    T0 = np.ones_like(xp)
    T1 = xp
    T2 = 2.0 * xp * xp - 1.0
    T3 = 4.0 * xp**3 - 3.0 * xp
    
    # Adaptive nonlinear terms
    # Clip shape params to avoid extreme values that cause numerical issues
    freq_tanh = np.clip(p[5], 0.1, 10.0)
    freq_sin = np.clip(p[6], 0.1, 10.0)
    gauss_width = np.clip(p[7], 0.1, 10.0)
    gauss_center = np.clip(p[8], -1.0, 1.0)
    tanh_scale = np.clip(p[9], 0.1, 5.0)
    
    # Basis functions
    b0 = T0
    b1 = T1
    b2 = T2
    b3 = T1 * np.tanh(freq_tanh * xp)
    b4 = np.exp(-gauss_width * (xp - gauss_center) ** 2)
    
    # Linear combination with weights p[0:5]
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
