import numpy as np
import time

# EVOLVE-BLOCK-START
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a hybrid basis of Chebyshev polynomials and 
    localized Gaussian RBFs. This combines the global stability of Chebyshev 
    polynomials with the local flexibility of RBFs to approximate the unknown 
    function.
    
    Basis functions (10 total):
    0: T_0(t) = 1 (Constant)
    1: T_1(t) = t (Linear)
    2: T_2(t) = 2t^2 - 1 (Quadratic)
    3: T_3(t) = 4t^3 - 3t (Cubic)
    4: T_4(t) = 8t^4 - 8t^2 + 1 (Quartic)
    5: T_5(t) = 16t^5 - 20t^3 + 5t (Quintic)
    6: RBF centered at -1.5, width 1.0
    7: RBF centered at 0.0, width 1.0
    8: RBF centered at 1.5, width 1.0
    9: RBF centered at 3.0, width 1.0
    
    Note: The RBFs are placed to cover the domain [-3, 3] effectively. 
    Chebyshev polynomials are mapped to t = x/3.
    """
    # Explicit deadline guard for safety, though computation is O(N) and fast.
    # The evaluator kills after 150s, but individual calls should be fast.
    # We don't need a complex time check inside this linear algebra call 
    # unless we were doing iterative fitting, which we are not (BFGS does that).
    
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    
    if N == 0:
        return np.array([])
        
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Pre-allocate basis matrix
    basis = np.empty((N, 10), dtype=np.float64)
    
    # --- Compute Chebyshev Polynomials T_0 to T_5 ---
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # Recurrence: T_{k+1}(t) = 2t * T_k(t) - T_{k-1}(t)
    # T_2(t)
    basis[:, 2] = 2.0 * t * basis[:, 1] - basis[:, 0]
    
    # T_3(t)
    basis[:, 3] = 2.0 * t * basis[:, 2] - basis[:, 1]
    
    # T_4(t)
    basis[:, 4] = 2.0 * t * basis[:, 3] - basis[:, 2]
    
    # T_5(t)
    basis[:, 5] = 2.0 * t * basis[:, 4] - basis[:, 3]
    
    # --- Compute Gaussian RBFs ---
    # Centers: -1.5, 0.0, 1.5, 3.0
    # Width (sigma): 1.0
    # This choice of centers and width aims to provide local flexibility 
    # across the domain, complementing the global polynomial trend.
    centers = np.array([-1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    # Compute squared distances: (x - c)^2
    # x shape: (N,), centers shape: (4,)
    # We can compute these in a loop to avoid temporary large arrays if N is huge,
    # but N=200 is small. Vectorized is fine.
    for i, c in enumerate(centers):
        diff = x - c
        sq_dist = diff * diff
        basis[:, 6 + i] = np.exp(-sq_dist / sigma2)
    
    # Linear combination: y = basis @ params
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if numerical issues arise (very unlikely)
        # Using the first few Chebyshev terms which are well-behaved
        return params[0] * basis[:, 0] + params[1] * basis[:, 1] + params[2] * basis[:, 2]
        
    return pred
# EVOLVE-BLOCK-END
