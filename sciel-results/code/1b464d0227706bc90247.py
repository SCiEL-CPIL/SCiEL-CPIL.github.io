"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear symbolic regression candidate using a Mixture of Gaussians.
    
    Strategy:
    The parent program used a linear basis expansion (p @ basis). This forces the 
    evaluator's BFGS optimizer to find the optimal linear combination of fixed basis 
    functions. If the target function is not well-approximated by the span of those 
    basis functions, or if the basis is ill-conditioned, the fit fails or is poor.
    
    This strategy uses a non-linear model: a sum of 5 Gaussian kernels.
    Model: y(x) = sum_{i=1 to 5} A_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2))
    
    Parameters (10 total):
    - params[0:5]  : Amplitudes (A_i)
    - params[5:10] : Log-widths (log_sigma_i). We use log to ensure sigma > 0.
    
    Wait, the evaluator fits `params` of length 10. 
    A mixture of 5 Gaussians has 3 params per kernel (A, mu, sigma) -> 15 params. 
    We only have 10.
    
    Alternative Non-Linear Model with 10 params:
    Let's use 2 Gaussian kernels + 4 linear terms? 
    Or 1 Gaussian + Polynomial?
    
    Let's look at the "Inspiration 1" which used RBFs and got a very high score (bad).
    The "Inspiration 3" got 385 (better than parent 74k).
    
    Let's try a model that is flexible but has exactly 10 params.
    
    Option A: Mixture of 2 Gaussians (6 params) + Quadratic Polynomial (4 params).
    y = A1*exp(-(x-mu1)^2/2s1^2) + A2*exp(-(x-mu2)^2/2s2^2) + c0 + c1*x + c2*x^2 + c3*x^3
    
    This covers:
    - Local bumps/peaks (Gaussians)
    - Global trend (Polynomial)
    
    This is a standard "additive model" often used in curve fitting.
    It is non-linear in params (due to exp and mu/sigma inside), so BFGS is required 
    and can potentially find a better local minimum than a linear least squares would, 
    provided the initial guess (all ones) isn't too far off.
    
    Initial guess analysis:
    params = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    If we map:
    p0, p1: A1, log_s1  (mu1 fixed? No, we need centers. We have 10 params.)
    
    Let's distribute 10 params:
    5 Gaussians is too many.
    2 Gaussians (A, mu, sigma) = 6 params.
    Remaining 4 params for a cubic polynomial: c0, c1, c2, c3.
    
    Total 10 params.
    
    Mapping:
    p0: A1 (Amplitude 1)
    p1: mu1 (Center 1)
    p2: log_sigma1 (Log-width 1)
    p3: A2 (Amplitude 2)
    p4: mu2 (Center 2)
    p5: log_sigma2 (Log-width 2)
    p6: c0 (Poly const)
    p7: c1 (Poly linear)
    p8: c2 (Poly quad)
    p9: c3 (Poly cubic)
    
    Initial values (all 1s):
    A1=1, mu1=1, log_s1=1 -> s1=e^1=2.7
    A2=1, mu2=1, log_s2=1 -> s2=e^1=2.7
    c0=1, c1=1, c2=1, c3=1
    
    This creates a specific initial curve. BFGS will adjust.
    
    Why this might beat the linear basis:
    1. Linear basis with fixed centers/frequencies might miss the target's specific
       frequency or center.
    2. Learnable centers (mu) and widths (sigma) allow the model to "move" to fit 
       local features.
    3. The polynomial part handles the global trend, preventing the Gaussians from 
       having to distort to fit the background.
    
    Robustness:
    - x in [-3, 3].
    - exp arguments: -(x-mu)^2 / (2*sigma^2).
    - If sigma is very small, exp becomes a spike. If sigma is very large, it's flat.
    - We use log_sigma to keep sigma > 0.
    - If BFGS drives sigma to 0 (log_sigma -> -inf), exp becomes 0 everywhere except 
      exactly at mu (measure zero). This is stable numerically if we clip.
    - We should clip log_sigma to avoid overflow/underflow issues in exp.
      Range [-10, 10] for log_sigma is safe (sigma in [4.5e-5, 22026]).
    
    Let's implement this.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has at least 10 elements, pad if necessary
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Extract parameters
    A1, mu1, log_s1 = p[0], p[1], p[2]
    A2, mu2, log_s2 = p[3], p[4], p[5]
    c0, c1, c2, c3 = p[6], p[7], p[8], p[9]
    
    # Clip log_sigma to prevent numerical instability
    # exp(-10) ~ 4.5e-5, exp(10) ~ 22026
    log_s1 = np.clip(log_s1, -10, 10)
    log_s2 = np.clip(log_s2, -10, 10)
    
    s1 = np.exp(log_s1)
    s2 = np.exp(log_s2)
    
    # Prevent division by zero if s becomes 0 (though clip prevents this)
    s1 = np.maximum(s1, 1e-8)
    s2 = np.maximum(s2, 1e-8)
    
    # Calculate Gaussian terms
    # Use stable exp: exp(-0.5 * ((x-mu)/sigma)^2)
    term1 = A1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
    term2 = A2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    
    # Calculate Polynomial term
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    pred = term1 + term2 + poly
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
