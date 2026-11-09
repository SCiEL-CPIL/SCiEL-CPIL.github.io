import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Orthogonal basis using Legendre polynomials on [-1, 1]
    # L0(t) = 1
    # L1(t) = t
    # L2(t) = 0.5 * (3t^2 - 1)
    # L3(t) = 0.5 * (5t^3 - 3t)
    
    # Precompute Legendre polynomials for x and y
    L0_x = np.ones_like(x)
    L1_x = x
    L2_x = 0.5 * (3.0 * x * x - 1.0)
    L3_x = 0.5 * (5.0 * x * x * x - 3.0 * x)
    
    L0_y = np.ones_like(y)
    L1_y = y
    L2_y = 0.5 * (3.0 * y * y - 1.0)
    L3_y = 0.5 * (5.0 * y * y * y - 3.0 * y)
    
    # Construct 10 basis functions from tensor products
    # Ordered by total degree to capture low-frequency components first
    term0 = L0_x * L0_y  # 1
    term1 = L1_x * L0_y  # x
    term2 = L0_x * L1_y  # y
    term3 = L2_x * L0_y  # 0.5(3x^2-1)
    term4 = L0_x * L2_y  # 0.5(3y^2-1)
    term5 = L1_x * L1_y  # xy
    term6 = L3_x * L0_y  # 0.5(5x^3-3x)
    term7 = L0_x * L3_y  # 0.5(5y^3-3y)
    term8 = L2_x * L1_y  # 0.5(3x^2-1)y
    term9 = L1_x * L2_y  # x0.5(3y^2-1)

    # Use matrix-vector multiplication for better numerical stability
    # and potentially faster evaluation
    basis_matrix = np.column_stack([
        term0, term1, term2, term3, term4,
        term5, term6, term7, term8, term9
    ])
    
    return basis_matrix @ p
