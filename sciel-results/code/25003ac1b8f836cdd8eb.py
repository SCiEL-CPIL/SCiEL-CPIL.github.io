"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a fixed linear spline (hat function) basis.
    
    We define 10 fixed knots uniformly spaced across the domain [-3, 3].
    The basis consists of 10 triangular "hat" functions centered at these knots.
    Each parameter in `params` corresponds to the height of the spline at the 
    corresponding knot.
    
    This approach is structurally different from global polynomials:
    1. Local: Each parameter only influences a local region (width ~1.2).
    2. Piecewise Linear: Can approximate non-smooth or oscillatory functions better than low-degree polynomials.
    3. Well-conditioned: The basis matrix is sparse and the columns are not highly correlated globally, 
       which helps BFGS converge reliably from the initial guess of all ones.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Define 10 fixed knots uniformly spaced in [-3, 3]
    # Knots: -3, -2.333, -1.666, ..., 2.333, 3
    # We use 10 knots to match the 10 parameters.
    # Note: Standard spline interpolation with N knots usually yields N-1 segments.
    # Here, we treat the 10 parameters as the values at 10 specific centers.
    # To ensure coverage of the entire domain, we place centers at:
    # c_i = -3 + (i + 0.5) * (6.0 / 10.0) for i in 0..9
    # This centers them within the 10 sub-intervals of width 0.6.
    # Alternatively, using knots at the boundaries and interior points is standard for B-splines.
    # Let's use a simple "hat" basis where each hat is supported on [c_i - w, c_i + w].
    
    n_params = 10
    domain_min = -3.0
    domain_max = 3.0
    span = domain_max - domain_min
    
    # Centers of the 10 hat functions
    # Spacing between centers
    step = span / n_params
    centers = domain_min + (np.arange(n_params) + 0.5) * step
    
    # Width of each hat function. 
    # If we make them touch, width = step. 
    # Let's make them overlap slightly or just touch to ensure continuous coverage.
    # A standard linear spline basis (like in finite elements) uses triangles.
    # If we define the basis as:
    # phi_i(x) = max(0, 1 - |x - c_i| / step)
    # Then they sum to 1 everywhere (partition of unity) if the domain is covered correctly.
    # With centers at midpoints of 10 bins of width 0.6, the support is [c_i - 0.6, c_i + 0.6].
    # This ensures that for any x in [-3, 3], the sum of basis functions is 1.
    # This is numerically very stable.
    
    width = step
    
    # Compute basis matrix T of shape (n_points, n_params)
    # T[i, j] = phi_j(x_i)
    n_points = x.size
    T = np.zeros((n_points, n_params), dtype=np.float64)
    
    # Vectorized computation of hat functions
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = np.abs(x[:, np.newaxis] - centers[np.newaxis, :])
    
    # phi_j(x) = max(0, 1 - |x - c_j| / width)
    # Clip negative values to 0
    T = np.maximum(0.0, 1.0 - diff / width)
    
    # Prediction is the linear combination
    return T @ params
# EVOLVE-BLOCK-END
