import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a mixed Chebyshev and RBF basis with scaled input.
    
    The model uses 10 parameters:
        f(x) = params[0] * T_0(z) + ... + params[5] * T_5(z)
              + params[6] * RBF_1(x) + ... + params[9] * RBF_4(x)
    
    where z = x/3.0 maps [-3,3] to [-1,1].
    
    Chebyshev polynomials provide a well-conditioned global basis for smooth trends,
    while RBF kernels provide local flexibility to capture deviations or features
    that a pure polynomial basis might miss. This hybrid approach maintains
    linearity in parameters, ensuring BFGS can reliably optimize the weights.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # 1. Scale x to [-1, 1] for Chebyshev basis
    z = x / 3.0
    
    # 2. Compute Chebyshev Polynomials T_0 to T_5
    # We use 6 Chebyshev terms for the global smooth component
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z**2 - 1.0
    T3 = 4.0 * z**3 - 3.0 * z
    T4 = 8.0 * z**4 - 8.0 * z**2 + 1.0
    T5 = 16.0 * z**5 - 20.0 * z**3 + 5.0 * z
    
    # Combine Chebyshev part: p[0]*T0 + ... + p[5]*T5
    cheb_part = (p[0] * T0 + 
                 p[1] * T1 + 
                 p[2] * T2 + 
                 p[3] * T3 + 
                 p[4] * T4 + 
                 p[5] * T5)
                 
    # 3. RBF Part
    # 4 RBFs to utilize the remaining 4 params (p[6] to p[9])
    # Centers: Symmetrically placed in [-3, 3]
    # Avoiding the exact boundaries to ensure coverage
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    
    # Fixed width for RBFs
    # A sigma of 1.0 provides a good balance between local and global influence
    # on the domain [-3, 3].
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (4,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # rbf_values shape: (N, 4)
    # p[6:10] shape: (4,)
    rbf_part = rbf_values @ p[6:10]
    
    result = cheb_part + rbf_part
    
    return result
