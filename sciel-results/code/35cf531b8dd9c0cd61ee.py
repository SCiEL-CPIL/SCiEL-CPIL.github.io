import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural change: Hybrid model combining a low-degree polynomial (for global trend)
    # with a set of adaptive Gaussian bumps (for local features).
    # This addresses the limitation of pure polynomial bases (Chebyshev/Legendre) which struggle
    # with sharp local variations, and pure RBFs which may struggle with global trends if centers
    # are fixed or poorly placed.
    
    # Parameter allocation (10 params total):
    # p0, p1, p2: Coefficients for polynomial part (degree 2)
    # mu1, log_sigma1, amp1: First Gaussian bump
    # mu2, log_sigma2, amp2: Second Gaussian bump
    # mu3, log_sigma3, amp3: Third Gaussian bump
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    
    mu1 = params[3]
    log_sigma1 = params[4]
    amp1 = params[5]
    
    mu2 = params[6]
    log_sigma2 = params[7]
    amp2 = params[8]
    
    mu3 = params[9]
    # We need one more param for log_sigma3 and amp3, but we only have 10.
    # Let's re-allocate to fit 3 Gaussians + Poly into 10 params?
    # Poly(3) + 2 Gaussians(6) = 9 params. 1 spare.
    # Or Poly(2) + 3 Gaussians(9) = 11 params. Too many.
    # Let's stick to Poly(2) + 2 Gaussians(6) = 8 params.
    # Remaining 2 params?
    
    # Let's try: Poly(3) [p0,p1,p2,p3] + 2 Gaussians [mu1, log_sig1, amp1, mu2, log_sig2, amp2] = 10 params.
    # This seems balanced. A cubic polynomial can capture a wide range of global behaviors.
    # Two Gaussians can capture two distinct local features.
    
    # Re-defining allocation:
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    mu1 = params[4]
    log_sigma1 = params[5]
    amp1 = params[6]
    
    mu2 = params[7]
    log_sigma2 = params[8]
    amp2 = params[9]
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Gaussian 1
    # Clip log_sigma to keep sigma in a reasonable range [0.1, 10]
    # log(0.1) ~ -2.3, log(10) ~ 2.3
    log_sigma1_clipped = np.clip(log_sigma1, -3.0, 2.3)
    sigma1 = np.exp(log_sigma1_clipped)
    z1 = (x - mu1) / sigma1
    gauss1 = amp1 * np.exp(-0.5 * z1**2)
    
    # Gaussian 2
    log_sigma2_clipped = np.clip(log_sigma2, -3.0, 2.3)
    sigma2 = np.exp(log_sigma2_clipped)
    z2 = (x - mu2) / sigma2
    gauss2 = amp2 * np.exp(-0.5 * z2**2)
    
    return poly + gauss1 + gauss2
