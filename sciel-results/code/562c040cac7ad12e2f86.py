"""Program for symbolic regression task using polynomial + Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model combining a low-degree polynomial and Gaussian RBFs.
    
    params[0:2]  -> coefficients for 1, x (linear trend)
    params[2:6]  -> amplitudes for 4 RBFs
    params[6:10] -> log-widths for 4 RBFs (sigma = exp(log_width))
    
    Centers are fixed at -2.25, -0.75, 0.75, 2.25.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # 1. Polynomial part: p[0] + p[1]*x
    # Using just linear to let RBFs do the heavy lifting for curvature?
    # Or maybe quadratic?
    # Let's stick to 2 poly params as planned to maximize RBF slots.
    poly_part = p[0] + p[1] * x
    
    # 2. RBF part
    # Fixed centers
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    
    # Log-widths, clipped for stability
    # If log_width is very negative, sigma is tiny. If very positive, sigma is huge.
    # Clip to [-4, 4] => sigma in [0.018, 54.6]
    log_widths = np.clip(p[6:10], -4.0, 4.0)
    widths = np.exp(log_widths)
    
    # Amplitudes
    amps = p[2:6]
    
    # Calculate RBF contributions
    # x shape (N,), centers shape (4,)
    # diff shape (N, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent: -diff^2 / (2 * width^2)
    # width shape (4,), broadcast to (N, 4)
    exponent = -0.5 * (diff ** 2) / (widths ** 2)
    
    # Gaussian values
    gaussians = np.exp(exponent)
    
    # Weighted sum
    rbf_part = np.sum(gaussians * amps[np.newaxis, :], axis=1)
    
    result = poly_part + rbf_part
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result

# EVOLVE-BLOCK-END
