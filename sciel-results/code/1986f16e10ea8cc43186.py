"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 10 parameters.
    
    We treat the 10 parameters as 5 fixed 2D centers (cx_i, cy_i) for i=0..4.
    The output is a weighted sum of Gaussian kernels evaluated at the distance
    from each input point (x, y) to each center.
    
    Since we only have 10 parameters and need to define the centers and weights,
    we can split them:
    - params[0:5] -> x-coordinates of centers
    - params[5:10] -> y-coordinates of centers
    
    But we also need weights. With only 10 parameters, we can't have separate
    weights and centers. 
    
    Alternative approach: Use a fixed set of centers and learn the weights.
    But the directive says "treated as fixed 1D centers". Let's re-read:
    "computing the Euclidean distance from each input point (x, y) to the 10 
    parameter values treated as fixed 1D centers"
    
    This is slightly ambiguous. If the centers are "fixed", they aren't learned.
    But `params` is the array of length 10 that is fit.
    
    Let's interpret it as: The 10 parameters define the model.
    A common simple RBF setup with 10 params:
    Use 5 centers. Each center has an x and y coordinate. That's 10 params for centers.
    Then we need weights. We don't have params left.
    
    Maybe the weights are fixed? Or maybe the architecture is different.
    
    Let's try a different interpretation:
    Use 10 fixed centers (pre-defined) and 10 learnable weights?
    But `params` length is 10. If centers are fixed, `params` can be the weights.
    
    Let's define 10 fixed centers in the [-1, 1] x [-1, 1] square.
    For example, a 3x3 grid plus one in the center? Or just 10 random fixed points?
    A 3x3 grid has 9 points. We can add the center (0,0) to make 10.
    
    Centers:
    (-1, -1), (0, -1), (1, -1)
    (-1,  0), (0,  0), (1,  0)
    (-1,  1), (0,  1), (1,  1)
    (0, 0) -> duplicate? No, let's just pick 10 distinct points.
    
    Actually, a simpler RBF model:
    y = sum_{i=1}^{10} w_i * exp(-gamma * ||(x,y) - c_i||^2)
    
    If c_i are fixed, w_i are the 10 parameters.
    This is a linear model in the parameters!
    BFGS will handle it, but it's essentially linear regression with a non-linear feature map.
    This is a very strong hypothesis space for smooth functions on a bounded domain.
    
    Let's define the 10 centers.
    To cover the domain [-1,1]x[-1,1] well:
    A 3x3 grid covers it, but has 9 points.
    Let's use:
    c0 = (-1.0, -1.0)
    c1 = ( 0.0, -1.0)
    c2 = ( 1.0, -1.0)
    c3 = (-1.0,  0.0)
    c4 = ( 0.0,  0.0)
    c5 = ( 1.0,  0.0)
    c6 = (-1.0,  1.0)
    c7 = ( 0.0,  1.0)
    c8 = ( 1.0,  1.0)
    c9 = ( 0.5,  0.5)  # Just an extra point to break symmetry or add capacity
    
    Or better, a 4x3 grid? No, 10 points.
    How about 5 points on x-axis and 5 on y-axis? No, 2D distance.
    
    Let's stick to the 3x3 grid + center (0,0) is already in the grid.
    Let's use:
    x_coords = [-1, -0.5, 0, 0.5, 1]
    y_coords = [-1, -0.5, 0, 0.5, 1]
    That's 25 points. Too many.
    
    Let's use 10 points:
    4 corners, 4 midpoints of edges, 1 center, 1 random?
    
    Actually, the "fixed 1D centers" phrase in the prompt is weird.
    "distance from each input point (x, y) to the 10 parameter values treated as fixed 1D centers"
    If centers are 1D, maybe it means |x - c_i|? But we have 2 inputs.
    Maybe it means the parameters ARE the centers, and they are 1D?
    But x and y are 2D.
    
    Let's assume the standard interpretation: 10 parameters are the weights of an RBF network with fixed centers.
    
    Centers:
    Let's use a 3x3 grid on [-1, 1] and one point at (0.5, 0.5) to make 10.
    Actually, 3x3 is 9.
    Let's just use 10 points spread out.
    
    cx = [-1, -1, -1, 0, 0, 1, 1, 1, 0, 0]
    cy = [-1, 0, 1, -1, 1, -1, 0, 1, -0.5, 0.5]
    
    Wait, let's just use a simple grid.
    np.linspace(-1, 1, 3) gives [-1, 0, 1].
    Meshgrid gives 9 points.
    We need 10.
    Let's add (0, 0) again? No, that's redundant.
    Let's add (0.5, 0.5).
    
    Gamma: The width of the Gaussian.
    If gamma is too small, the function is smooth (global).
    If gamma is too large, the function is spiky (local).
    Since we don't have a param for gamma, we must fix it.
    A reasonable gamma for domain [-1,1] is around 1.0 to 5.0.
    Let's try gamma = 2.0.
    
    equation:
    pred = sum_{i=0}^{9} params[i] * exp(-gamma * ((x - cx[i])**2 + (y - cy[i])**2))
    
    This is linear in params. BFGS will converge quickly.
    It can fit any function that is a linear combination of these 10 Gaussians.
    This is a much richer space than a linear model (x, y, 1).
    """
    
    # Define 10 fixed centers
    # 3x3 grid + (0.5, 0.5)
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5]
    ])
    
    # x and y are arrays of shape (N,)
    # centers is shape (10, 2)
    
    # Compute squared distances
    # x shape (N, 1), centers[:, 0] shape (10, 1) -> diff (N, 10)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Gamma (inverse variance)
    gamma = 2.0
    
    # RBF values: shape (N, 10)
    rbf_values = np.exp(-gamma * dist_sq)
    
    # params shape (10,)
    # pred shape (N,)
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
