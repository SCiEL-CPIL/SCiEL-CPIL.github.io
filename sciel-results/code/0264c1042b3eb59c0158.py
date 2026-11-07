# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    - 4 Chebyshev polynomials (global trend, well-conditioned)
    - 6 RBF terms with fixed centers (local detail/adaptivity)
    
    Total: 10 parameters (linear in params, so BFGS converges well)
    
    Chebyshev basis is mapped to [-1, 1] from [0.1, 6] for stability.
    RBF centers are evenly spaced across the domain with sigma=1.0.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # --- Chebyshev Basis (4 terms) ---
    # Map x from [0.1, 6] to [-1, 1]
    x_min, x_max = 0.1, 6.0
    z = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    # T0 = 1, T1 = z, T2 = 2z^2 - 1, T3 = 4z^3 - 3z
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * z - 1.0
    T3 = 4.0 * z * z * z - 3.0 * z
    
    # --- RBF Basis (6 terms) ---
    # Evenly spaced centers across [0.1, 6]
    n_centers = 6
    centers = np.linspace(x_min + (x_max - x_min) / (2 * n_centers), 
                          x_max - (x_max - x_min) / (2 * n_centers), n_centers)
    sigma = 1.0
    
    # Compute RBF values: exp(-(x - c_j)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (n, 6)
    
    # Stack basis: shape (n, 10)
    # Columns 0-3: Chebyshev, Columns 4-9: RBF
    basis = np.column_stack([T0, T1, T2, T3, rbf_values])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
