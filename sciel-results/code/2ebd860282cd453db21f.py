"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute standardized basis functions.
    Returns a matrix of shape (n, 10) where each column is a standardized basis function.
    
    Basis functions (10 total):
    0: 1 (bias)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: x^5
    6: exp(x)
    7: exp(-x)
    8: log(x + 1)
    9: sin(x)
    
    Each column is standardized (zero mean, unit variance) to improve conditioning.
    """
    n = len(x)
    basis = np.zeros((n, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    basis[:, 5] = x ** 5
    basis[:, 6] = np.exp(x)
    basis[:, 7] = np.exp(-x)
    basis[:, 8] = np.log(x + 1.0)
    basis[:, 9] = np.sin(x)
    
    # Standardize each column: zero mean, unit variance
    for j in range(10):
        col = basis[:, j]
        mean = np.mean(col)
        std = np.std(col)
        if std > 1e-12:
            basis[:, j] = (col - mean) / std
        else:
            basis[:, j] = col - mean
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 standardized basis functions.
    
    The parameters are weights for each standardized basis function.
    Since the basis is standardized, the design matrix is well-conditioned,
    allowing BFGS to converge reliably to the global least-squares solution.
    
    Note: We need to account for the standardization offset.
    If basis_j(x) = (f_j(x) - mean_j) / std_j, then:
    sum_i params[i] * basis_i(x) = sum_i params[i] * (f_i(x) - mean_i) / std_i
    = sum_i (params[i] / std_i) * f_i(x) - sum_i (params[i] * mean_i) / std_i
    
    To make this a proper linear-in-parameters model that can represent any
    function in the span of {f_0, ..., f_9}, we should NOT standardize in a way
    that changes the parameter interpretation. Instead, let's use the raw basis
    but with careful scaling to avoid ill-conditioning.
    
    Actually, the key insight is: for linear-in-parameters models, BFGS starting
    from all ones needs to find the correct linear combination. The issue with
    unscaled basis is that different basis functions have vastly different scales,
    making the Hessian ill-conditioned.
    
    A better approach: use the raw basis functions but let the optimizer handle it.
    The problem is that with raw basis, some columns (like x^5 or exp(x)) have
    very large values, making the problem ill-conditioned.
    
    Alternative: use a fixed, well-scaled set of basis functions.
    Let's use: 1, x, x^2, x^3, x^4, x^5, exp(x/2), exp(-x/2), log(x+1), sin(x)
    and scale them to have similar magnitudes over [0.1, 6].
    """
    n = len(x)
    
    # Compute raw basis functions
    f0 = np.ones(n)
    f1 = x
    f2 = x ** 2
    f3 = x ** 3
    f4 = x ** 4
    f5 = x ** 5
    f6 = np.exp(x / 2.0)
    f7 = np.exp(-x / 2.0)
    f8 = np.log(x + 1.0)
    f9 = np.sin(x)
    
    # Scale each basis function to have unit norm (L2) to improve conditioning
    # This makes the design matrix columns have similar magnitudes
    scales = np.array([
        np.sqrt(np.mean(f0 ** 2)),
        np.sqrt(np.mean(f1 ** 2)),
        np.sqrt(np.mean(f2 ** 2)),
        np.sqrt(np.mean(f3 ** 2)),
        np.sqrt(np.mean(f4 ** 2)),
        np.sqrt(np.mean(f5 ** 2)),
        np.sqrt(np.mean(f6 ** 2)),
        np.sqrt(np.mean(f7 ** 2)),
        np.sqrt(np.mean(f8 ** 2)),
        np.sqrt(np.mean(f9 ** 2)),
    ])
    
    # Avoid division by zero
    scales[scales < 1e-12] = 1.0
    
    # Build the design matrix with scaled basis
    # We want: prediction = sum_i params[i] * (f_i(x) / scale_i)
    # This is equivalent to: prediction = sum_i (params[i] / scale_i) * f_i(x)
    # So the effective coefficients for the raw basis are params[i] / scale_i
    
    basis_scaled = np.column_stack([
        f0 / scales[0],
        f1 / scales[1],
        f2 / scales[2],
        f3 / scales[3],
        f4 / scales[4],
        f5 / scales[5],
        f6 / scales[6],
        f7 / scales[7],
        f8 / scales[8],
        f9 / scales[9],
    ])
    
    # Ensure finite values
    basis_scaled = np.nan_to_num(basis_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Linear combination
    result = basis_scaled @ params
    
    return result
# EVOLVE-BLOCK-END
