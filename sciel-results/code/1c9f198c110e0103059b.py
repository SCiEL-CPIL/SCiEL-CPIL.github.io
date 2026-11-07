"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale Gaussian RBF network with 7 kernels.
    
    Params layout (10 total):
    - params[0:7]: centers c_i for 7 Gaussian kernels
    - params[7:10]: amplitudes a_i for the first 3 kernels (last 4 kernels share amplitude 1.0)
    
    Wait, we need exactly 10 params. Let me redesign:
    - params[0:5]: centers c_i for 5 kernels
    - params[5:8]: log-widths w_i for 3 of the kernels (kernel 4 and 5 use fixed widths)
    - params[8:10]: amplitudes a_0, a_1 for first 2 kernels (rest derived)
    
    Actually, let's keep it simpler but more flexible:
    - params[0:5]: centers c_i
    - params[5:10]: amplitudes a_i
    
    But make the widths learnable via a softplus transform of a fixed offset.
    No, we only have 10 params. Let me use:
    - params[0:5]: centers
    - params[5:10]: amplitudes
    And use a mix of fixed widths that are well-spaced.
    
    Better structural change: Use a hybrid of local RBFs and global terms.
    params[0:6]: centers for 6 RBF kernels
    params[6:8]: amplitudes for first 2 RBFs
    params[8:10]: coefficients for x and x^2 global terms
    
    f(x) = sum_{i=1}^{6} a_i * exp(-((x-c_i)^2)/(2*sigma_i^2)) + p8*x + p9*x^2
    
    where sigma_i alternates between 0.5 and 1.5 for multi-scale coverage,
    and a_i for i>=3 are fixed to 1.0 (only first 2 amplitudes are learned).
    """
    centers = params[:6]
    amps_0 = params[6]
    amps_1 = params[7]
    c_lin = params[8]
    c_quad = params[9]
    
    # Multi-scale fixed widths: alternate narrow and wide
    sigmas = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
    
    # Amplitudes: first two learned, rest fixed at 1.0
    amps = np.array([amps_0, amps_1, 1.0, 1.0, 1.0, 1.0])
    
    # Compute RBF values: shape (n_points, 6)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernels with per-kernel widths
    rbf_values = np.exp(-0.5 * (diff / sigmas[np.newaxis, :]) ** 2)
    
    # Linear combination of RBFs
    result = rbf_values @ amps
    
    # Add global linear and quadratic terms
    result = result + c_lin * x + c_quad * x**2
    
    return result
# EVOLVE-BLOCK-END
