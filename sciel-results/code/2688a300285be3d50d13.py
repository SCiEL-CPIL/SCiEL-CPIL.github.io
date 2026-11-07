import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a fixed-center Gaussian RBF basis.
    
    10 RBF centers are spread across the 2D (x, v) training domain.
    Each of the 10 fitted parameters is the amplitude of one RBF.
    This gives a flexible, smooth, bounded nonlinear model that is
    well-conditioned for BFGS (features are bounded in [0,1], not
    near-collinear when centers are spaced).
    """
    # Fixed RBF centers spread across the training domain
    # x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
    centers_x = np.array([
        -0.45,  0.0,   0.40,
        -0.25,  0.25,
        -0.50,  0.45,
        -0.10,  0.15,
        0.30
    ])
    centers_v = np.array([
        -0.10,  0.0,   0.15,
        -0.08,  0.18,
        -0.12,  0.20,
        0.05,  -0.05,
        0.10
    ])
    # Fixed widths: moderate overlap for smooth interpolation
    # x scale ~ 1.1, v scale ~ 0.46
    sigmas = np.array([
        0.35, 0.30, 0.35,
        0.30, 0.30,
        0.35, 0.35,
        0.25, 0.25,
        0.30
    ])

    # Compute RBF values: exp(-((x - cx)^2 / (2*sigma_x^2) + (v - cv)^2 / (2*sigma_v^2)))
    # Use separate widths for x and v to match their different scales
    sigma_x = sigmas * 1.0       # x-width
    sigma_v = sigmas * 0.45      # v-width (v range is ~0.46, x range is ~1.1)

    # Efficient computation: x[:, None] - centers_x[None, :]
    dx = x[:, None] - centers_x[None, :]
    dv = v[:, None] - centers_v[None, :]

    # RBF matrix: (n_samples, 10)
    rbf = np.exp(-0.5 * (dx**2 / sigma_x[None, :]**2 + dv**2 / sigma_v[None, :]**2))

    # Linear combination of RBF amplitudes
    return rbf @ params
