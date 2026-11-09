import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic Polynomial + 2 Gaussian Bumps.
    Parameters:
    p0, p1, p2, p9: Coefficients for 1, x, x^2, x^3
    p3, p4, p5: Center, log-width, Amplitude for Gaussian 1
    p6, p7, p8: Center, log-width, Amplitude for Gaussian 2
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Polynomial part: p0 + p1*x + p2*x^2 + p9*x^3
    poly = p0 + p1 * x + p2 * x**2 + p9 * x**3
    
    # Gaussian 1
    # Use exp(p4) for width to ensure positivity and numerical stability
    # Clip log-width before exp to prevent overflow/underflow
    log_width1 = np.clip(p4, -3.0, 3.0)
    width1 = np.exp(log_width1)
    # Ensure minimum width to avoid division by zero
    width1 = np.maximum(width1, 0.01)
    
    # Compute exponent with clipping to avoid overflow in exp
    z1 = (x - p3) / width1
    z1_sq = z1 * z1
    # Clip the squared distance to prevent exp(-large) from underflowing to 0 too quickly
    # or causing issues, though exp(-large) is safe, we want to avoid inf in intermediate steps if width is tiny
    # Actually, if width is small, z is large, z^2 is large, exp(-z^2/2) is 0. That's fine.
    # But if width is huge, z is small, exp is ~1.
    # The main risk is if p4 is very large, width is huge, but we clipped log_width.
    # If p4 is very small, width is small. z can be large.
    # Let's just compute it directly.
    gauss1 = p5 * np.exp(-0.5 * z1_sq)
    
    # Gaussian 2
    log_width2 = np.clip(p7, -3.0, 3.0)
    width2 = np.exp(log_width2)
    width2 = np.maximum(width2, 0.01)
    
    z2 = (x - p6) / width2
    z2_sq = z2 * z2
    gauss2 = p8 * np.exp(-0.5 * z2_sq)
    
    return poly + gauss1 + gauss2
