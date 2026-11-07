"""Rewritten program: RBF basis with fixed centers for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed set of Radial Basis Functions (RBFs)
    and a constant term, weighted by the 10 fitted parameters.
    
    Basis:
    - 9 Gaussian RBFs centered on a 3x3 grid within [-1, 1]x[-1, 1]
    - 1 Constant term
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Define centers for the 3x3 grid
    # Spacing 0.6 covers [-0.6, 0.6] which is good for [-1, 1] domain
    # We could go wider, e.g., [-0.8, 0, 0.8]
    cx = [-0.7, 0.0, 0.7]
    cy = [-0.7, 0.0, 0.7]
    
    # Width of the Gaussian kernels
    # Needs to be large enough to overlap, small enough to be local
    sigma = 0.6
    
    # Pre-compute distances for each center
    # x and y are arrays of shape (N,)
    
    f = np.zeros_like(x)
    
    # Map parameters to centers
    # p0: (-0.7, -0.7)
    # p1: (-0.7, 0.0)
    # p2: (-0.7, 0.7)
    # p3: ( 0.0, -0.7)
    # p4: ( 0.0, 0.0)
    # p5: ( 0.0, 0.7)
    # p6: ( 0.7, -0.7)
    # p7: ( 0.7, 0.0)
    # p8: ( 0.7, 0.7)
    # p9: Constant
    
    centers = [
        (cx[0], cy[0]), (cx[0], cy[1]), (cx[0], cy[2]),
        (cx[1], cy[0]), (cx[1], cy[1]), (cx[1], cy[2]),
        (cx[2], cy[0]), (cx[2], cy[1]), (cx[2], cy[2])
    ]
    
    for i, (xc, yc) in enumerate(centers):
        # Compute squared distance
        dx = x - xc
        dy = y - yc
        dist_sq = dx**2 + dy**2
        # Gaussian kernel
        kernel = np.exp(-dist_sq / (2.0 * sigma**2))
        f += params[i] * kernel
        
    # Add constant term
    f += p9
    
    return f
# EVOLVE-BLOCK-END
