import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: if we have already exceeded a safe portion of the limit, 
    # return a reasonable fallback. However, this function is called 
    # frequently by the optimizer, so it must be fast.
    # The 120s limit is for the whole evaluation (including fit).
    # Our function should be < 1ms.
    
    # Multi-scale RBF Basis
    # 1 Bias
    # 5 Wide RBFs (sigma=1.0) at: (-1,-1), (-1,1), (1,-1), (1,1), (0,0)
    # 4 Narrow RBFs (sigma=0.4) at: (-0.5,-0.5), (-0.5,0.5), (0.5,-0.5), (0.5,0.5)
    
    # Extract params
    p0 = params[0]
    pw = params[1:6]  # 5 wide weights
    pn = params[6:10] # 4 narrow weights
    
    # Define centers
    # Wide centers
    cx_w = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    cy_w = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    # Narrow centers
    cx_n = np.array([-0.5, -0.5, 0.5, 0.5])
    cy_n = np.array([-0.5, 0.5, -0.5, 0.5])
    
    # Sigmas
    sigma_w = 1.0
    sigma_n = 0.4
    
    # Compute Wide RBFs
    # x, y are (N,)
    # cx_w, cy_w are (5,)
    # We want (N, 5) matrix of distances squared
    dx_w = x[:, np.newaxis] - cx_w[np.newaxis, :]
    dy_w = y[:, np.newaxis] - cy_w[np.newaxis, :]
    dist_sq_w = dx_w**2 + dy_w**2
    rbf_w = np.exp(-dist_sq_w / (2 * sigma_w**2))
    
    # Compute Narrow RBFs
    dx_n = x[:, np.newaxis] - cx_n[np.newaxis, :]
    dy_n = y[:, np.newaxis] - cy_n[np.newaxis, :]
    dist_sq_n = dx_n**2 + dy_n**2
    rbf_n = np.exp(-dist_sq_n / (2 * sigma_n**2))
    
    # Prediction
    pred = p0 + np.sum(rbf_w * pw, axis=1) + np.sum(rbf_n * pn, axis=1)
    
    return pred
