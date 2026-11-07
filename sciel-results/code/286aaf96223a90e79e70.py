"""
Strategy: Learnable Weighted Gaussian Mixture Model (GMM).
Instead of a fixed linear basis (polynomials/trig), we use 10 Gaussian bumps.
The parameters are:
- p[0:5]: Centers of the first 5 Gaussians
- p[5:10]: Centers of the next 5 Gaussians
- p[10:15]: Widths (sigma) of the first 5 Gaussians
- p[15:20]: Weights (amplitudes) of the first 5 Gaussians
- p[20:25]: Widths (sigma) of the next 5 Gaussians
- p[25:30]: Weights (amplitudes) of the next 5 Gaussians

Wait, the task says params is an array of length 10.
"fits params, an array of length 10"
So I cannot use 30 parameters. I must stick to 10 parameters.

Let's re-read carefully: "fits params, an array of length 10".
Okay, so I have exactly 10 floats.

How to make a powerful model with only 10 parameters?
Option A: 5 Gaussians.
- 2 centers, 1 width, 1 weight per Gaussian? That's 4 params * 5 = 20 params. Too many.
- 1 center, 1 width, 1 weight? 3 * 5 = 15. Still too many.
- 1 center, 1 weight? Width fixed? 2 * 5 = 10.
  - Fixed width might be too rigid.
  
Option B: 2 Gaussians.
- 3 params per Gaussian (center, width, weight) * 2 = 6 params.
- Remaining 4 params? Maybe a polynomial background?
  - 2 Gaussians (6 params) + Linear background (2 params: slope, intercept) + Quadratic term (1 param) + Cubic term (1 param). Total 10.
  
Option C: 1 Gaussian + Polynomial.
- 1 Gaussian (3 params: c, w, a) + 7 degree polynomial?
  - Polynomial degree 7 is 8 coeffs. 3+8=11. Too many.
  - 1 Gaussian (3 params) + Degree 6 polynomial (7 coeffs) = 10 params.
  - This allows a global polynomial trend plus one local bump.
  
Option D: Multiple Gaussians with shared structure?
- What if I use 5 Gaussians but constrain them?
- Or just 2 very flexible Gaussians?
- Actually, a common "universal approximator" with few params is hard.
- But wait, the evaluator uses BFGS starting from all ones.
- If I use a GMM with 2 components:
  - c1, s1, w1, c2, s2, w2 (6 params)
  - Plus a linear term: m, b (2 params)
  - Plus a quadratic term: q (1 param)
  - Plus a cubic term: r (1 param)
  - Total 10.
  
This seems like a good hybrid. It can fit local bumps (Gaussians) and global trends (polynomials).
The previous program used fixed basis polynomials and trigs. The issue with fixed basis is that if the function has a sharp feature not aligned with the basis, it fails. A learnable Gaussian can move to where the feature is.

Let's refine the parameter mapping for 2 Gaussians + Polynomial background.
Params:
0: c1 (center 1)
1: s1 (width 1)
2: w1 (weight 1)
3: c2 (center 2)
4: s2 (width 2)
5: w2 (weight 2)
6: a (linear coeff)
7: b (const)
8: q (quadratic coeff)
9: r (cubic coeff)

Expression:
y = w1 * exp(-((x-c1)/s1)^2) + w2 * exp(-((x-c2)/s2)^2) + a*x + b + q*x^2 + r*x^3

Numerical stability:
- s1, s2 must be positive. If BFGS drives them to 0, exp(-inf) is 0, which is fine, but gradient might vanish or be unstable.
- Better to use softplus or exp for widths?
- If I use exp(s1) for width, then s1 can be any real number. This ensures width > 0.
- Let's use:
  s1_eff = np.exp(params[1])
  s2_eff = np.exp(params[4])
  
This adds non-linearity to the parameter space, which BFGS handles well if the function is smooth.

Let's check the starting point: all ones.
c1=1, s1_exp=1 -> s1=e^1=2.718, w1=1
c2=1, s2_exp=1 -> s2=e^1=2.718, w2=1
a=1, b=1, q=1, r=1
Initial prediction: 1*exp(-((x-1)/2.7)^2) + 1*exp(-((x-1)/2.7)^2) + x + 1 + x^2 + x^3
= 2*exp(...) + x^3+x^2+x+1.
This is a reasonable starting point.

Is this materially different?
Yes. The current program uses a fixed linear basis. This program uses a non-linear, adaptive basis (Gaussian centers and widths are learned, not fixed). This allows the model to "move" features to match the data, rather than just scaling fixed features.

Let's implement this.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    
    # Extract parameters
    c1 = params[0]
    s1_log = params[1]
    w1 = params[2]
    
    c2 = params[3]
    s2_log = params[4]
    w2 = params[5]
    
    a = params[6]
    b = params[7]
    q = params[8]
    r = params[9]
    
    # Ensure widths are positive and stable
    # Using exp ensures s > 0. Clamping log-width prevents overflow in exp for very large positive s_log
    # However, if s_log is very large, exp(s_log) is huge, exp(-((x-c)/huge)^2) -> 1.
    # If s_log is very small (negative), exp(s_log) -> 0, exp(-((x-c)/0)^2) -> 0 (except at x=c where it's 1).
    # To avoid numerical issues with division by zero or tiny numbers, we can clamp the effective width.
    # But standard exp is usually fine within reasonable ranges.
    # Let's add a small epsilon to the squared denominator to avoid div by 0 if s is extremely small.
    
    s1 = np.exp(s1_log)
    s2 = np.exp(s2_log)
    
    # Calculate Gaussians
    # Using np.where or safe division?
    # (x - c) / s
    # If s is 0, we get inf. exp(-inf) is 0.
    # If x == c and s == 0, we get 0/0 = nan.
    # To be safe, let's ensure s is at least 1e-6.
    s1_safe = np.maximum(s1, 1e-6)
    s2_safe = np.maximum(s2, 1e-6)
    
    g1 = np.exp(-((x - c1) / s1_safe) ** 2)
    g2 = np.exp(-((x - c2) / s2_safe) ** 2)
    
    # Polynomial background
    poly = a * x + b + q * x**2 + r * x**3
    
    return w1 * g1 + w2 * g2 + poly
