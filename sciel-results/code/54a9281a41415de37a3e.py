import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Convert to float64 array
    x = np.asarray(x, dtype=np.float64)
    
    # Handle potential edge cases
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev stability
    # t = (2*x - (min+max)) / (max-min)
    # min=0.1, max=6.0
    # t = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Initialize array for basis functions
    # shape: (len(x), 10)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
        
    basis = np.zeros((n, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
