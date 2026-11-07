import numpy as np

def _halton2d(n, base1=2, base2=3):
    """Generate n points in 2D using Halton sequences for quasi-random coverage."""
    points = []
    for i in range(1, n + 1):
        # 1D Halton in base b
        def halton_1d(idx, base):
            f = 1.0
            r = 0.0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r
        h1 = halton_1d(i, base1)
        h2 = halton_1d(i, base2)
        # Map [0,1) to [-1,1]
        points.append((2.0 * h1 - 1.0, 2.0 * h2 - 1.0))
    return np.array(points)

# Precompute RBF centers and widths
# Use 4 RBFs with fixed geometry (centers from Halton, fixed width)
# Parameters: p[0..3] = RBF amplitudes, p[4..9] = polynomial coefficients
# Total: 4 RBF + 6 poly = 10 params

_RBF_CENTERS = _halton2d(4)  # 4 centers in [-1,1]^2
_RBF_WIDTH = 0.5  # Fixed Gaussian width

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse RBF + polynomial basis for symbolic regression.
    
    Parameters (length 10):
    p[0..3]: Amplitudes of 4 fixed-geometry Gaussian RBFs
    p[4]:    Intercept
    p[5]:    Coeff for x
    p[6]:    Coeff for y
    p[7]:    Coeff for x^2
    p[8]:    Coeff for y^2
    p[9]:    Coeff for x*y
    
    The RBF centers and width are fixed (deterministic Halton grid),
    making the model linear in all 10 parameters. BFGS will find the
    global optimum for this linear model.
    """
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y

    # RBF terms: Gaussian bumps with fixed centers and width
    # rbf_i = exp(-||[x,y] - c_i||^2 / (2 * width^2))
    rbf_terms = np.zeros_like(x)
    for i in range(4):
        cx, cy = _RBF_CENTERS[i]
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        rbf_val = np.exp(-dist_sq / (2.0 * _RBF_WIDTH * _RBF_WIDTH))
        rbf_terms += params[i] * rbf_val

    # Construct prediction
    pred = (params[4]
            + params[5] * x
            + params[6] * y
            + params[7] * x2
            + params[8] * y2
            + params[9] * xy
            + rbf_terms)

    return pred
