import numpy as np

# Pre-compute statistics for basis functions over x in [-2, 2]
# We use a fine grid to approximate the mean and std
_x_grid = np.linspace(-2, 2, 10000)

def _compute_stats(func):
    vals = func(_x_grid)
    mu = np.mean(vals)
    sigma = np.std(vals)
    if sigma < 1e-6:
        sigma = 1.0
    return mu, sigma

# Define raw basis functions
# 0: Constant (not standardized, kept as 1)
# 1: x
# 2: x^2
# 3: x^3
# 4: x^4
# 5: sin(x)
# 6: cos(x)
# 7: sin(2x)
# 8: cos(2x)
# 9: exp(-x^2)

_stats = [None] * 10
_stats[1] = _compute_stats(lambda x: x)
_stats[2] = _compute_stats(lambda x: x**2)
_stats[3] = _compute_stats(lambda x: x**3)
_stats[4] = _compute_stats(lambda x: x**4)
_stats[5] = _compute_stats(lambda x: np.sin(x))
_stats[6] = _compute_stats(lambda x: np.cos(x))
_stats[7] = _compute_stats(lambda x: np.sin(2*x))
_stats[8] = _compute_stats(lambda x: np.cos(2*x))
_stats[9] = _compute_stats(lambda x: np.exp(-x**2))

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    
    # Compute standardized basis values
    # b[0] is 1
    b0 = np.ones_like(x)
    
    # b[1] to b[9] are standardized
    b1 = (x - _stats[1][0]) / _stats[1][1]
    b2 = (x**2 - _stats[2][0]) / _stats[2][1]
    b3 = (x**3 - _stats[3][0]) / _stats[3][1]
    b4 = (x**4 - _stats[4][0]) / _stats[4][1]
    b5 = (np.sin(x) - _stats[5][0]) / _stats[5][1]
    b6 = (np.cos(x) - _stats[6][0]) / _stats[6][1]
    b7 = (np.sin(2*x) - _stats[7][0]) / _stats[7][1]
    b8 = (np.cos(2*x) - _stats[8][0]) / _stats[8][1]
    b9 = (np.exp(-x**2) - _stats[9][0]) / _stats[9][1]
    
    # Combine
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
