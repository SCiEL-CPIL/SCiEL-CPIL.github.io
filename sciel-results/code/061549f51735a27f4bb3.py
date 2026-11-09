"""
Strategy: Non-linear parametric model.
Instead of a fixed linear basis (y = sum p_i * b_i(x)), we use a model where
parameters define the structure of the function. This allows BFGS to adjust
frequencies, widths, and amplitudes, effectively performing symbolic regression
over a family of common functions (polynomials, trig, sigmoids).

The model:
y = A1*tanh(k1*x) + A2*sin(w1*x) + A3*cos(w2*x) + A4*x^2 + A5*x + A6
    + A7 * sigmoid(k2*(x - c1)) + A8 * sigmoid(k3*(x - c2))
    + A9 * exp(-0.5*((x-c3)/s1)^2) + A10 * exp(-0.5*((x-c4)/s2)^2)

Parameters (10 total):
p[0]=A1, p[1]=k1, p[2]=A2, p[3]=w1, p[4]=A3, p[5]=w2, p[6]=A4, p[7]=A5, p[8]=A6, p[9]=...
Wait, I need 10 params. Let's simplify to a robust set of 10.

Let's use:
1. Poly part: a0 + a1*x + a2*x^2 + a3*x^3
2. Trig part: a4*sin(w*x) + a5*cos(w*x)
3. Sigmoid part: a6 * sigmoid(k*x)
4. Gaussian part: a7 * exp(-0.5*((x-c)/s)^2)

Params:
p[0]=a0, p[1]=a1, p[2]=a2, p[3]=a3
p[4]=a4, p[5]=w
p[6]=a5
p[7]=a6, p[8]=k
p[9]=a7 ... wait, I need c and s for Gaussian. That's 3 params.
Total: 4 (poly) + 2 (trig amp/freq) + 1 (trig cos amp) + 2 (sigmoid) + 3 (gaussian) = 12. Too many.

Let's stick to 10.
1. Poly: a0 + a1*x + a2*x^2
2. Trig: a3*sin(w*x) + a4*cos(w*x)
3. Sigmoid: a5 * sigmoid(k*x)
4. Rational/Gaussian-like: a6 * exp(-x^2) * cos(p7*x) ? No, let's keep it simple.

Alternative 10-param model:
y = p0 + p1*x + p2*x^2 + p3*x^3
  + p4*sin(p5*x) + p6*cos(p7*x)
  + p8*tanh(p9*x)

This uses all 10 params. It covers polynomials, single-frequency sines/cosines, and a centered sigmoid.
BFGS will optimize all 10.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear model:
    y = p0 + p1*x + p2*x^2 + p3*x^3
      + p4*sin(p5*x) + p6*cos(p7*x)
      + p8*tanh(p9*x)
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety: if params are not finite, return zeros (or handle gracefully)
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Trig part
    # Handle potential large frequencies causing numerical issues in sin/cos?
    # sin/cos are bounded, so it's safe.
    trig = p4 * np.sin(p5 * x) + p6 * np.cos(p7 * x)
    
    # Sigmoid part
    # tanh is bounded [-1, 1], safe.
    sig = p8 * np.tanh(p9 * x)
    
    out = poly + trig + sig
    
    # Ensure finite output
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        
    return out
# EVOLVE-BLOCK-END
