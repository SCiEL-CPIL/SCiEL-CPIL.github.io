"""Structural change: Local Linear RBF with adaptive centers, 3 centers in 3D space."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Linear RBF model for nonlinear oscillator.
    
    Strategy:
    Instead of a global polynomial/trigonometric basis, use a local linear RBF approach.
    This is a different design family: local approximation vs global basis.
    
    We place 3 RBF centers in the (x, v) space (ignoring t for locality, but including
    t in the local linear part). Each center has:
    - An intercept (bias)
    - A slope in x
    - A slope in v
    - A slope in t (or sin/cos of t)
    
    With 3 centers and 4 params each = 12 params, but we only have 10.
    
    Let's use 2 centers with 5 params each = 10 params.
    Or 3 centers with 3 params each = 9, plus 1 global bias = 10.
    
    Actually, let's try: 3 RBF centers, each with (weight, x_slope, v_slope) = 9 params,
    plus 1 global bias = 10 params.
    
    The RBF centers are fixed at strategic locations in the input range.
    Input ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    
    Centers at:
    c1 = (x=0, v=0) - center
    c2 = (x=-0.15, v=-0.25) - lower-left
    c3 = (x=0.15, v=0.25) - upper-right
    
    For each center, compute a Gaussian RBF: exp(-dist^2 / (2*sigma^2))
    Then the local linear prediction at center i is: w_i * (1 + a_i * x + b_i * v) * RBF_i
    
    Wait, that's not quite right. Let me think again.
    
    Local Linear RBF: For each center, we have a local linear function.
    The output is a weighted sum of local linear functions, where weights are RBFs.
    
    prediction = sum_i [ RBF_i(x,v) * (w0_i + w1_i * x + w2_i * v) ]
    
    With 3 centers and 3 params each (w0, w1, w2) = 9 params. Plus 1 global bias = 10.
    
    Actually, let's just use 3 centers with 3 params each = 9, and add 1 more term.
    Or use 2 centers with 5 params each.
    
    Let me go with: 3 RBF centers, each contributing a local linear term.
    Params: for each center i (i=1,2,3): [intercept_i, slope_x_i, slope_v_i] = 9 params
    Plus 1 global bias = 10 params.
    
    RBF width: fixed, not a parameter. Use sigma = 0.15 (roughly 1/3 of the range).
    """
    
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Fixed RBF centers in (x, v) space
    # Input ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    centers = np.array([
        [0.0, 0.0],       # center
        [-0.15, -0.25],   # lower-left
        [0.15, 0.25]      # upper-right
    ])
    
    sigma = 0.15
    sigma2 = 2.0 * sigma * sigma
    
    # Compute RBF weights for each center
    # dx = x - center_x, dv = v - center_v
    # rbf = exp(-(dx^2 + dv^2) / sigma2)
    
    rbf1 = np.exp(-((x - centers[0, 0])**2 + (v - centers[0, 1])**2) / sigma2)
    rbf2 = np.exp(-((x - centers[1, 0])**2 + (v - centers[1, 1])**2) / sigma2)
    rbf3 = np.exp(-((x - centers[2, 0])**2 + (v - centers[2, 1])**2) / sigma2)
    
    # Local linear terms
    # Center 1: params[0] + params[1]*x + params[2]*v
    # Center 2: params[3] + params[4]*x + params[5]*v
    # Center 3: params[6] + params[7]*x + params[8]*v
    # Global bias: params[9]
    
    local1 = rbf1 * (params[0] + params[1] * x + params[2] * v)
    local2 = rbf2 * (params[3] + params[4] * x + params[5] * v)
    local3 = rbf3 * (params[6] + params[7] * x + params[8] * v)
    
    # Add a term involving t to capture time-dependence
    # Use sin(t) and cos(t) but we've used all 10 params.
    # Let's adjust: use 2 centers with 4 params each = 8, plus 2 for t.
    
    # Revised: 2 RBF centers, each with (intercept, slope_x, slope_v, slope_t) = 8 params
    # Plus 2 global terms: bias and sin(t) coefficient
    
    # Actually, let's keep it simple and just add t-dependence to the local linear terms.
    # Use sin(t) as the time feature instead of t directly.
    
    st = np.sin(t)
    
    # Revised params layout:
    # Center 1: params[0]=intercept, params[1]=slope_x, params[2]=slope_v
    # Center 2: params[3]=intercept, params[4]=slope_x, params[5]=slope_v
    # Center 3: params[6]=intercept, params[7]=slope_x, params[8]=slope_v
    # Global: params[9] = bias
    
    # But we need time dependence. Let me rethink.
    
    # Alternative: use t in the RBF centers too? No, t range is [30,50], too large.
    
    # Let's use: prediction = global_bias + sum_i RBF_i(x,v) * (w0_i + w1_i*x + w2_i*v + w3_i*sin(t))
    # That's 4 params per center. With 2 centers = 8, plus 2 global = 10.
    
    # Let me redo with 2 centers.
    
    centers2 = np.array([
        [0.0, 0.0],
        [0.0, 0.0]  # same center but different... no, that's bad.
    ])
    
    # Better: 2 centers at different locations.
    # c1 = (0, 0), c2 = (0.1, 0.1)
    
    c1x, c1v = 0.0, 0.0
    c2x, c2v = 0.1, 0.1
    
    rbf1 = np.exp(-((x - c1x)**2 + (v - c1v)**2) / sigma2)
    rbf2 = np.exp(-((x - c2x)**2 + (v - c2v)**2) / sigma2)
    
    # Each center has 4 params: intercept, slope_x, slope_v, slope_sin_t
    # Center 1: params[0:4]
    # Center 2: params[4:8]
    # Global: params[8] = bias, params[9] = cos(t) coefficient
    
    local1 = rbf1 * (params[0] + params[1] * x + params[2] * v + params[3] * st)
    local2 = rbf2 * (params[4] + params[5] * x + params[6] * v + params[7] * st)
    
    result = params[8] + params[9] * np.cos(t) + local1 + local2
    
    return result

# EVOLVE-BLOCK-END
