import numpy as np
import time

# Hard deadline guard: return safely before the 120 s limit.
# The equation function is called repeatedly during BFGS optimization.
# We track time to ensure we don't exceed the limit if the function is called many times.
# However, since the function is stateless per call (except for global time), 
# and the evaluator runs BFGS which calls this many times, we just need to be fast.
# The 120s limit is for the whole evaluation. We don't need a complex guard inside equation
# unless we do heavy computation. This basis is O(N) and very fast.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev + RBF Basis.
    
    Strategy:
    - Use 3 low-order Chebyshev polynomials (T0, T1, T2) for global smooth trends.
    - Use 7 Gaussian RBFs centered at evenly spaced points in [0.1, 6] for local flexibility.
    - This allows the model to capture both global structure and local deviations.
    - The design matrix is linear in parameters, so BFGS optimizes a convex problem.
    - The RBF centers are fixed; only the weights (params) are optimized.
    
    Parameters (10 total):
    p[0]: Coefficient for T_0(t)
    p[1]: Coefficient for T_1(t)
    p[2]: Coefficient for T_2(t)
    p[3] to p[9]: Weights for 7 RBFs centered at x_c = 0.1 + (i+0.5)*5.9/7 for i=0..6
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)

    # 1. Global Chebyshev terms (3 terms)
    min_x = 0.1
    max_x = 6.0
    t = 2.0 * (x - min_x) / (max_x - min_x) - 1.0
    
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    T0 = np.ones_like(x)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    
    # 2. Local RBF terms (7 terms)
    # Centers evenly spaced in [0.1, 6]
    # n_rbf = 7
    # centers[i] = min_x + (i + 0.5) * (max_x - min_x) / 7
    n_rbf = 7
    centers = min_x + (np.arange(n_rbf) + 0.5) * (max_x - min_x) / n_rbf
    
    # Width of the RBFs. 
    # The distance between centers is (max_x - min_x) / n_rbf = 5.9 / 7 ≈ 0.843
    # A sigma of ~0.3-0.5 provides good local coverage without too much overlap.
    # Let's use sigma = 0.4
    sigma = 0.4
    
    # Compute RBF values: exp(-((x - center)^2) / (2 * sigma^2))
    # Shape: (N, n_rbf)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / (2.0 * sigma**2))
    
    # 3. Combine
    # result = p[0]*T0 + p[1]*T1 + p[2]*T2 + sum(p[3+i] * rbf_values[:, i])
    result = p[0] * T0 + p[1] * T1 + p[2] * T2
    result += rbf_values @ p[3:10]
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
