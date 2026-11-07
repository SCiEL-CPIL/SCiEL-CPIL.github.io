"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed RBF centers in scaled space (inputs centered at 1.25, scaled by 0.75 to [-1,1])
# 3 RBFs + 7 polynomial terms = 10 params
_RBF_CENTERS = np.array([
    [0.0, 0.0, 0.0],
    [-0.5, -0.5, -0.5],
    [0.5, 0.5, 0.5]
])
_RBF_WIDTH = 0.8  # fixed width in scaled space

def _compute_basis(x, y, z):
    """Compute basis matrix: 3 RBF columns + 7 polynomial columns = 10 columns."""
    n = x.size
    xs = (x - 1.25) / 0.75
    ys = (y - 1.25) / 0.75
    zs = (z - 1.25) / 0.75
    
    # RBF features (3 columns)
    rbf = np.zeros((n, 3))
    for i in range(3):
        dx = xs - _RBF_CENTERS[i, 0]
        dy = ys - _RBF_CENTERS[i, 1]
        dz = zs - _RBF_CENTERS[i, 2]
        r2 = dx**2 + dy**2 + dz**2
        rbf[:, i] = np.exp(-r2 / (_RBF_WIDTH ** 2))
    
    # Polynomial features (7 columns)
    poly = np.column_stack([
        np.ones(n),
        xs,
        ys,
        zs,
        xs**2,
        ys**2,
        zs**2
    ])
    
    return np.hstack([rbf, poly])


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    
    if params.shape[0] != 10:
        # Fallback: just return a constant
        return np.full(n, params[0] if params.size > 0 else 0.0)
    
    basis = _compute_basis(x, y, z)
    return basis @ params
# EVOLVE-BLOCK-END
