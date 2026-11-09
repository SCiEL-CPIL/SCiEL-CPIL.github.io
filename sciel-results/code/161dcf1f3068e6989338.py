# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a sparse Gaussian RBF network
    combined with a global quadratic polynomial trend.
    
    The RBF centers are placed on a grid over the input domain to capture
    local nonlinear stiffness and damping variations. The polynomial trend
    captures global behavior. All basis functions are designed to be well-
    conditioned for BFGS optimization starting from ones.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to approximately [-1, 1] for better conditioning
    # x range ~ [-0.57, 0.53], center ~ -0.02, half-range ~ 0.55
    # v range ~ [-0.18, 0.28], center ~ 0.05, half-range ~ 0.23
    x_center = -0.02
    x_half = 0.55
    v_center = 0.05
    v_half = 0.23
    
    X = (x - x_center) / x_half  # approximately [-1, 1]
    V = (v - v_center) / v_half  # approximately [-1, 1]
    
    # Global quadratic polynomial trend (3 terms)
    b1 = np.ones_like(X)
    b2 = X
    b3 = X * X + V * V  # radial quadratic
    
    # Sparse RBF network with 7 Gaussian kernels
    # Centers chosen to cover the input domain
    # Using a mix of positions in (X, V) space
    
    # Define RBF centers in scaled coordinates
    centers = np.array([
        [0.0, 0.0],    # center
        [-0.5, 0.0],   # left
        [0.5, 0.0],    # right
        [0.0, -0.5],   # down
        [0.0, 0.5],    # up
        [-0.4, 0.4],   # corner
        [0.4, -0.4],   # corner
    ])
    
    # RBF width parameter - controls locality
    # Smaller sigma = more local, larger sigma = more global
    sigma = 0.6
    
    # Compute RBF basis functions
    b4 = np.zeros_like(X)
    b5 = np.zeros_like(X)
    b6 = np.zeros_like(X)
    b7 = np.zeros_like(X)
    b8 = np.zeros_like(X)
    b9 = np.zeros_like(X)
    b10 = np.zeros_like(X)
    
    for i in range(7):
        dx = X - centers[i, 0]
        dv = V - centers[i, 1]
        r2 = dx * dx + dv * dv
        rbf = np.exp(-r2 / (2 * sigma * sigma))
        
        if i == 0:
            b4 = rbf
        elif i == 1:
            b5 = rbf
        elif i == 2:
            b6 = rbf
        elif i == 3:
            b7 = rbf
        elif i == 4:
            b8 = rbf
        elif i == 5:
            b9 = rbf
        elif i == 6:
            b10 = rbf
    
    # Combine: 3 polynomial + 7 RBF = 10 basis functions
    y = (p[0] * b1 +
         p[1] * b2 +
         p[2] * b3 +
         p[3] * b4 +
         p[4] * b5 +
         p[5] * b6 +
         p[6] * b7 +
         p[7] * b8 +
         p[8] * b9 +
         p[9] * b10)
         
    return y
# EVOLVE-BLOCK-END
