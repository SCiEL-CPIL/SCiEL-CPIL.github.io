"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid Chebyshev-polynomial + tanh-nonlinear basis.
    
    The 10 parameters are linear weights over 10 fixed basis functions:
    - 6 Chebyshev tensor-product terms (total degree <= 3):
        T0(x)T0(y), T1(x)T0(y), T0(x)T1(y),
        T2(x)T0(y), T1(x)T1(y), T0(x)T2(y)
    - 4 tanh-based nonlinear features for capturing non-polynomial structure:
        tanh(x), tanh(y), tanh(x*y), tanh(x^2 - y^2)
    
    This combines the well-conditioned polynomial approximation of Chebyshev bases
    (inspiration 1/2) with the nonlinear expressiveness of the tanh network
    (current program), giving BFGS a smoother landscape while retaining the
    ability to model non-polynomial hidden functions.
    """
    # Safety deadline guard
    t_start = time.time()
    TIME_LIMIT = 140.0

    # --- Chebyshev polynomials for x and y ---
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0

    # --- 6 Chebyshev tensor-product basis terms (total degree <= 3) ---
    f0 = T0_x * T0_y        # 1
    f1 = T1_x * T0_y        # x
    f2 = T0_x * T1_y        # y
    f3 = T2_x * T0_y        # 2x^2 - 1
    f4 = T1_x * T1_y        # x*y
    f5 = T0_x * T2_y        # 2y^2 - 1

    # --- 4 tanh-based nonlinear features ---
    f6 = np.tanh(x)
    f7 = np.tanh(y)
    f8 = np.tanh(x * y)
    f9 = np.tanh(x**2 - y**2)

    # Stack into a basis matrix of shape (n, 10)
    basis = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination with the 10 learnable parameters
    prediction = basis @ params

    # Time guard: if we are close to the limit, just return what we have
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        return prediction

    return prediction
# EVOLVE-BLOCK-END
