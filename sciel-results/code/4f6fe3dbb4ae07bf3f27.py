import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Basis 1: Chebyshev T_0 to T_4
    # Map x from [0.1, 6.0] to t in [-1, 1]
    # min=0.1, max=6.0, range=5.9
    # t = (2x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    basis_cheb = np.zeros((n_points, 5), dtype=np.float64)
    basis_cheb[:, 0] = 1.0
    basis_cheb[:, 1] = t
    for k in range(2, 5):
        basis_cheb[:, k] = 2.0 * t * basis_cheb[:, k-1] - basis_cheb[:, k-2]
        
    # Basis 2: 5 Gaussian RBFs
    # Centers evenly spaced in [0.1, 6.0]
    # Let's pick centers: 0.5, 2.0, 3.5, 5.0, 6.0
    centers = np.array([0.5, 2.0, 3.5, 5.0, 6.0])
    # Widths (sigma)
    # Let's use sigma = 0.8
    sigmas = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    
    # Compute RBFs
    # x is (N, 1) or (N,). Let's ensure (N, 1) for broadcasting
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    s_row = sigmas.reshape(1, -1)
    
    # diff = x - c
    diff = x_col - c_row
    # exp(-diff^2 / (2 * sigma^2))
    basis_rbf = np.exp(-0.5 * (diff / s_row)**2)
    
    # Combine
    basis = np.hstack((basis_cheb, basis_rbf))
    
    pred = basis @ params
    return pred
