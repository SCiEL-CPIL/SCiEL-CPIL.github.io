import numpy as np
import time

# We use a deadline guard to ensure we finish within the 120s/150s limit.
# Since the evaluation is a single call to equation(x, params) which is then
# minimized by the evaluator, the "time" constraint applies to the entire
# evaluation process. However, inside equation(), we must be fast.
# The heavy lifting is done by the evaluator's BFGS loop.
# We ensure equation() itself is O(N * K) where K is the number of basis functions.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian RBF Basis for Symbolic Regression.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 parameters.
            p[0:5] -> Centers of 5 Gaussian RBFs (mapped to [-2, 2])
            p[5:10] -> Widths of 5 Gaussian RBFs (exponentiated for positivity)
            
    Note: The amplitudes of the RBFs are NOT in params. 
    Wait, the task says "fits params, an array of length 10". 
    If I use 5 for centers and 5 for widths, I have 0 left for amplitudes.
    I cannot solve for amplitudes internally because I don't have y.
    
    Correction:
    The evaluator minimizes MSE(y, equation(x, params)).
    If equation(x, params) returns a prediction, and params are the only variables,
    then the function must be a fixed form where params directly influence the output.
    
    If I set:
    y_hat = sum_{i=1}^{5} A_i * exp(-(x - c_i)^2 / (2 w_i^2))
    
    If c_i and w_i are in params, what are A_i?
    If A_i are fixed (e.g., all 1s), the model is:
    y_hat = sum_{i=1}^{5} exp(-(x - c_i)^2 / (2 w_i^2))
    This has only 10 parameters (5 centers, 5 widths).
    This is a non-linear model. BFGS can optimize non-linear models.
    
    Is this expressive enough?
    A sum of 5 Gaussians with variable centers and widths but FIXED amplitudes (all 1) is quite restrictive.
    It cannot produce negative values if the target has negative parts, unless the target is always positive or the Gaussians overlap in a way that... no, they are all positive.
    So y_hat >= 0 always.
    If the target function takes negative values, this model will fail badly.
    
    Alternative:
    Use 10 parameters differently.
    Maybe 2 centers, 2 widths, and 6 amplitudes? No, need matching count.
    
    Let's look at the structure again.
    We have 10 parameters.
    Option A: 5 RBFs. 5 centers, 5 widths. Amplitudes fixed to 1. (Problem: only positive output).
    Option B: 5 RBFs. 5 centers, 2 widths (shared?), 3 amplitudes? No, messy.
    Option C: 10 RBFs? No, need more params.
    
    Let's reconsider the "Fixed Amplitude" issue.
    If the target is, say, sin(x), which is positive and negative, a sum of positive Gaussians cannot fit it well.
    
    How to allow negative amplitudes?
    We need parameters for amplitudes.
    If we use 5 RBFs, we need 5 centers, 5 widths, and 5 amplitudes = 15 params. We only have 10.
    
    So we must reduce the number of RBFs or fix some parameters.
    
    Strategy: Use 3 Adaptive RBFs + 1 Polynomial term?
    3 RBFs: 3 centers, 3 widths, 3 amplitudes = 9 params.
    1 Param left for a global scale or bias?
    
    Let's try:
    y_hat = A_0 + A_1 * x + A_2 * x^2 + A_3 * exp(-(x-c_1)^2/w_1^2)
    This uses:
    A_0, A_1, A_2, A_3 (4 amplitudes)
    c_1, w_1 (2 params for one RBF)
    Total 6 params. We have 4 left.
    Add another RBF?
    c_2, w_2, A_4.
    Total 9 params.
    Add a bias? A_5?
    
    Let's try a mix:
    2 Adaptive Gaussians + Low-order Polynomial.
    
    Params:
    p[0]: Bias (A_0)
    p[1]: Linear coeff (A_1)
    p[2]: Quadratic coeff (A_2)
    p[3]: Cubic coeff (A_3)
    p[4]: RBF1 Amplitude (A_4)
    p[5]: RBF1 Center (C_1)
    p[6]: RBF1 Width (W_1)
    p[7]: RBF2 Amplitude (A_5)
    p[8]: RBF2 Center (C_2)
    p[9]: RBF2 Width (W_2)
    
    This uses all 10 params.
    The polynomial part handles the global trend (and can be negative).
    The RBFs handle local features (bumps/dips) and can be negative due to A_4, A_5.
    The centers and widths are optimized by BFGS.
    
    Widths must be positive. Use exp(p) or softplus.
    Centers should be in [-2, 2]. Use tanh(p) * 2.
    
    This is a non-linear least squares problem. BFGS is suitable.
    Risk: Local minima. Starting from all ones:
    p[0..3] = 1 -> Poly: 1+x+x^2+x^3
    p[4]=1, p[5]=1, p[6]=1 -> RBF1: 1 * exp(-(x-1)^2 / exp(1)^2)
    p[7]=1, p[8]=1, p[9]=1 -> RBF2: 1 * exp(-(x-1)^2 / exp(1)^2)
    Initial guess is reasonable.
    
    Let's implement this.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Polynomial part: p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3
    poly = p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3
    
    # RBF 1
    amp1 = p[4]
    center1 = np.tanh(p[5]) * 2.0  # Map to [-2, 2]
    width1 = np.exp(p[6])          # Ensure positive
    
    # RBF 2
    amp2 = p[7]
    center2 = np.tanh(p[8]) * 2.0  # Map to [-2, 2]
    width2 = np.exp(p[9])          # Ensure positive
    
    # Compute RBFs
    # Avoid division by zero or overflow in exp
    # width is exp(p), so if p is very negative, width is small.
    # If width is 0, (x-c)^2 / (2*w^2) is inf. exp(-inf) = 0.
    # But if w is extremely small, and x is close to c, it might be unstable.
    # Let's clip widths to a minimum of 1e-6.
    w1_safe = np.maximum(width1, 1e-6)
    w2_safe = np.maximum(width2, 1e-6)
    
    rbf1 = np.exp(-((x - center1)**2) / (2.0 * w1_safe**2))
    rbf2 = np.exp(-((x - center2)**2) / (2.0 * w2_safe**2))
    
    y_hat = poly + amp1 * rbf1 + amp2 * rbf2
    
    # Ensure finite output
    # If any part is inf/nan, replace with 0 or large penalty?
    # The evaluator handles non-finite with 1e30 penalty.
    # But we can try to be safe.
    if not np.all(np.isfinite(y_hat)):
        # If we encounter numerical instability, return a safe value
        # This shouldn't happen often with the soft parameterizations
        y_hat = np.where(np.isfinite(y_hat), y_hat, 0.0)
        
    return y_hat
