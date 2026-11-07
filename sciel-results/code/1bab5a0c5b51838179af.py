import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is in [-3, 3]
    # Approach: Chebyshev Series Approximation
    # This is a different family from Fourier series.
    # Chebyshev polynomials are orthogonal on [-1, 1] and bounded,
    # providing better numerical conditioning for BFGS than monomials.
    
    # Scale x to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = u
        for k in range(2, 10):
            T[k, :] = 2.0 * u * T[k-1, :] - T[k-2, :]
            
    # Linear combination
    y = np.sum(params[:, np.newaxis] * T, axis=0)
    
    return y
