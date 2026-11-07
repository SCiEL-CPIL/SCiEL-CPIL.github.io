import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Learnable Gaussian Mixture (4 components) + Linear Baseline.
    
    The 10 parameters are allocated as follows:
    - p[0]: Linear baseline slope (m)
    - p[1]: Linear baseline intercept (b)
    - p[2:6]: Centers (c1, c2, c3, c4) for 4 Gaussian components
    - p[6:10]: Widths (w1, w2, w3, w4) for 4 Gaussian components
    
    Amplitudes are not free parameters; they are implicitly determined by the
    BFGS optimizer adjusting the other parameters to minimize MSE. However, to
    allow for varying magnitudes, we can scale the Gaussian contributions.
    Wait, if amplitudes are fixed to 1.0, we can't change the height.
    The prompt says params is an array of length 10.
    
    Let's re-allocate:
    - p[0]: Baseline constant (b)
    - p[1:5]: Centers (c1, c2, c3, c4)
    - p[5:9]: Widths (w1, w2, w3, w4)
    - p[9]: Global amplitude scale for Gaussians? Or separate amplitudes?
    
    If we have 4 Gaussians, we need 4 centers, 4 widths, and 4 amplitudes. That's 12 params.
    We only have 10.
    
    Option A: 3 Gaussians (3c, 3w, 3a) + 1 baseline + 1 linear slope = 10 params.
    Option B: 4 Gaussians (4c, 4w) + 2 baseline (slope, intercept) = 10 params. 
             In this case, amplitudes are fixed to 1. This might be too restrictive.
    Option C: 2 Gaussians (2c, 2w, 2a) + Polynomial of degree 4 (5 coeffs) = 10 params.
             This combines local flexibility with global trend.
    
    Let's go with Option C: 2 Learnable Gaussians + Degree 4 Polynomial.
    Params:
    - p[0]: c1 (center 1)
    - p[1]: w1 (width 1)
    - p[2]: a1 (amplitude 1)
    - p[3]: c2 (center 2)
    - p[4]: w2 (width 2)
    - p[5]: a2 (amplitude 2)
    - p[6]: d0 (poly const)
    - p[7]: d1 (poly linear)
    - p[8]: d2 (poly quadratic)
    - p[9]: d3 (poly cubic)
    
    Wait, BFGS starting from all ones.
    If centers start at 1.0, they are on the right side of [-2, 2].
    Widths start at 1.0.
    Amplitudes start at 1.0.
    Polynomial coeffs start at 1.0.
    
    This allows the model to fit a global trend (polynomial) and local bumps (Gaussians).
    The polynomial handles the general shape, while Gaussians handle deviations.
    This is structurally different from a pure polynomial because the "kinks" or "bumps"
    can move and change shape, whereas polynomial coefficients only change the global curve shape.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Extract parameters
    c1 = params[0]
    w1 = params[1]
    a1 = params[2]
    c2 = params[3]
    w2 = params[4]
    a2 = params[5]
    d0 = params[6]
    d1 = params[7]
    d2 = params[8]
    d3 = params[9]
    
    # Ensure widths are positive to avoid division by zero or negative variance
    # Using softplus or abs? Abs is non-differentiable at 0 but BFGS might handle it poorly.
    # Better: w = 1.0 + 0.1 * param? No, param starts at 1.
    # Let's use w = np.exp(0.1 * w_param) to ensure positivity and smoothness?
    # If w_param=1, w=exp(0.1)~1.1.
    # Or simply w = 0.1 * np.abs(w_param) + 0.1?
    # Let's stick to a simple transformation that is differentiable and keeps scale reasonable.
    # w_eff = 0.5 + 0.5 * np.tanh(w_param) ? No, that bounds it.
    # Let's use w_eff = np.exp(0.5 * w_param) / 10.0? 
    # If w_param=1, w_eff = exp(0.5)/10 ~ 0.065. Too small.
    # If w_param=1, let's just use w_eff = 0.1 * (w_param**2 + 1.0).
    # If w_param=1, w_eff = 0.2.
    # If w_param=5, w_eff = 1.3.
    # This ensures w > 0.
    
    w1_eff = 0.1 * (w1**2 + 1.0)
    w2_eff = 0.1 * (w2**2 + 1.0)
    
    # Calculate Gaussians
    # y = a * exp(-(x-c)^2 / (2*w^2))
    g1 = a1 * np.exp(-((x - c1)**2) / (2.0 * w1_eff**2))
    g2 = a2 * np.exp(-((x - c2)**2) / (2.0 * w2_eff**2))
    
    # Calculate Polynomial part (Degree 3 to save a param? No, we have 4 params left)
    # y_poly = d0 + d1*x + d2*x^2 + d3*x^3
    # Scale x for polynomial stability? x in [-2, 2]. x^3 is [-8, 8]. Fine.
    y_poly = d0 + d1 * x + d2 * (x**2) + d3 * (x**3)
    
    return y_poly + g1 + g2
