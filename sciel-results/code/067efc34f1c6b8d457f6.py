import numpy as np

# Precompute standardization constants for the mixed basis on x ~ U[0.1, 6]
# We use a fine grid to approximate the moments
_x_grid = np.linspace(0.1, 6.0, 10000)
_basis_funcs = [
    lambda x: np.ones_like(x),
    lambda x: x,
    lambda x: x**2,
    lambda x: x**3,
    lambda x: x**4,
    lambda x: x**5,
    lambda x: np.log(x),
    lambda x: np.exp(-x),
    lambda x: np.sin(x),
    lambda x: np.cos(x)
]

_means = []
_stds = []
for f in _basis_funcs:
    vals = f(_x_grid)
    m = np.mean(vals)
    s = np.std(vals)
    if s < 1e-8:
        s = 1.0
    _means.append(m)
    _stds.append(s)

_means = np.array(_means)
_stds = np.array(_stds)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a mixed basis of polynomial, logarithmic, exponential, and trigonometric
    functions. This diverse set of functions allows the model to capture a wide
    range of target function behaviors (trends, decay, oscillation) with high
    fidelity. The basis is standardized to improve the conditioning of the
    optimization problem for BFGS.
    """
    n_points = x.shape[0]
    
    # Compute the basis functions
    # We do this in a vectorized way for efficiency
    basis = np.empty((n_points, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x**2
    basis[:, 3] = x**3
    basis[:, 4] = x**4
    basis[:, 5] = x**5
    basis[:, 6] = np.log(x)
    basis[:, 7] = np.exp(-x)
    basis[:, 8] = np.sin(x)
    basis[:, 9] = np.cos(x)
    
    # Standardize the basis columns
    # (basis - mean) / std
    basis_std = (basis - _means) / _stds
    
    # Linear combination
    return basis_std @ params
