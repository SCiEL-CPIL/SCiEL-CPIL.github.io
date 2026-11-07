import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bilinear Chebyshev polynomial basis.
    The basis consists of products of low-degree Chebyshev polynomials T_i(x) * T_j(x).
    This representation allows for efficient capture of multiplicative interactions
    and higher-order features while maintaining linearity in the parameters.
    """
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev definition
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Define 10 basis functions as products of low-degree Chebyshev polynomials
    # We select a mix of pure powers and mixed products to span a rich space
    # Note: These are not all linearly independent, but BFGS can handle some redundancy
    # if the condition number is not too high. We pick a set that covers degrees 0-5.
    
    basis = np.empty((10, len(x)))
    
    # 0: Constant
    basis[0] = T0 * T0  # 1
    
    # 1: Linear
    basis[1] = T1 * T0  # t
    
    # 2: Quadratic (Cheb)
    basis[2] = T2 * T0  # 2t^2 - 1
    
    # 3: Cubic (Cheb)
    basis[3] = T3 * T0  # 4t^3 - 3t
    
    # 4: Quartic (Cheb)
    basis[4] = T4 * T0  # 8t^4 - 8t^2 + 1
    
    # 5: Quadratic (Monomial-like from product)
    basis[5] = T1 * T1  # t^2
    
    # 6: Cubic (Mixed)
    basis[6] = T2 * T1  # t(2t^2 - 1) = 2t^3 - t
    
    # 7: Quartic (Mixed)
    basis[7] = T3 * T1  # t(4t^3 - 3t) = 4t^4 - 3t^2
    
    # 8: Quartic (Squared Cheb)
    basis[8] = T2 * T2  # (2t^2 - 1)^2 = 4t^4 - 4t^2 + 1
    
    # 9: Quintic (Mixed)
    basis[9] = T3 * T2  # (4t^3 - 3t)(2t^2 - 1) = 8t^5 - 10t^3 + 3t
    
    # Linear combination
    prediction = np.sum(params[:, np.newaxis] * basis, axis=0)
    
    return prediction
