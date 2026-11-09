import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Use Chebyshev polynomials of the first kind mapped to
    the domain [0.1, 6] as the basis functions. This replaces the mixed
    polynomial/trig/exp basis with a single, well-conditioned orthogonal-like
    basis that is specifically designed for uniform approximation on a bounded
    interval. This addresses the conditioning issues of high-degree monomials
    and provides a more uniform coverage of the function space over the
    specified domain, which should help BFGS converge more reliably to a
    good local minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    # t = (2 * (x - x_min) / (x_max - x_min)) - 1
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    basis = np.zeros((len(t), 10))
    basis[:, 0] = T0
    basis[:, 1] = T1
    
    for k in range(2, 10):
        Tk = 2.0 * t * T1 - T0
        basis[:, k] = Tk
        T0, T1 = T1, Tk
        
    # Linear combination
    return basis @ params
