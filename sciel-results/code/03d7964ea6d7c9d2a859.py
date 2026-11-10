import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Chebyshev Polynomial Regression with Scaled Input
    # 1. Scale input x from [-2, 2] to [-1, 1] to improve conditioning.
    # 2. Use Chebyshev polynomials of the first kind as the basis.
    #    These are bounded in [-1, 1] on the interval [-1, 1], avoiding
    #    the ill-conditioning of high-degree monomials (like x^9 which is ~512 at x=2).
    # 3. Use all 10 parameters for coefficients T_0 through T_9.
    # This creates a well-conditioned linear model, allowing BFGS to converge
    # reliably to the global optimum for the Chebyshev basis, which can
    # approximate smooth functions very efficiently (spectral convergence).
    
    # Scale x to [-1, 1]
    x_scaled = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    T = np.zeros((10, x.shape[0]), dtype=np.float64)
    T[0, :] = 1.0
    if x.shape[0] > 0:
        T[1, :] = x_scaled
        for n in range(2, 10):
            T[n, :] = 2.0 * x_scaled * T[n-1, :] - T[n-2, :]
            
    # The prediction is the linear combination of these basis functions
    # y = sum(p[i] * T[i, :])
    
    y = np.dot(p, T)
    
    return y
