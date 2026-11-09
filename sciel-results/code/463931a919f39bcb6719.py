import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 9 centers in a 3x3 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-1.0, 0.0, 1.0])
    cx, cy = np.meshgrid(xs, ys, indexing='ij')
    centers = np.column_stack((cx.ravel(), cy.ravel()))
    
    # x, y are shape (N,)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    gamma = 2.0
    rbf_values = np.exp(-gamma * dist_sq)
    
    # params[0:9] are weights, params[9] is bias
    pred = rbf_values @ params[:9] + params[9]
    
    return pred
