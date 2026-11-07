import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: RBF basis (3x3 grid, gamma=5) + linear terms + tanh nonlinearity.
    
    The 10 parameters are linear weights on 10 basis functions:
      0-8: RBF kernels at 3x3 grid points (-1,0,1) x (-1,0,1)
      9:  constant term (allows global offset)
    
    The RBF basis provides local, smooth interpolation capability.
    The constant term ensures the model can represent a global bias.
    
    This is a linear model in the parameters, so BFGS should converge well.
    The RBF basis is well-suited for smooth functions on [-1,1]^2.
    """
    # Pre-computed centers for 3x3 grid
    _grid = np.array([-1.0, 0.0, 1.0])
    _cx_grid, _cy_grid = np.meshgrid(_grid, _grid, indexing='ij')
    _cx_flat = _cx_grid.flatten()  # shape (9,)
    _cy_flat = _cy_grid.flatten()  # shape (9,)
    
    _GAMMA = 5.0
    
    # Compute RBF features
    dx = x[:, None] - _cx_flat[None, :]  # shape (N, 9)
    dy = y[:, None] - _cy_flat[None, :]  # shape (N, 9)
    dist2 = dx**2 + dy**2
    rbf = np.exp(-_GAMMA * dist2)  # shape (N, 9)
    
    # Build feature matrix: [rbf_0, ..., rbf_8, 1]
    ones = np.ones_like(x)
    features = np.column_stack((rbf, ones))  # shape (N, 10)
    
    # Linear combination
    prediction = features @ params
    
    return prediction
