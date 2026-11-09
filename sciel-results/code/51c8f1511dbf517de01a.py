"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill


def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Piecewise-Linear (PWL) interpolation over a 2x2 grid of the domain [-1, 1]^2.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64)
    
    # 2x2 grid vertices in x and y
    # x: -1, 1  -> indices 0, 1
    # y: -1, 1  -> indices 0, 1
    
    # Compute barycentric coordinates (linear weights) for each point
    # For x in [-1, 1]:
    # w_x0 = (1 - x) / 2  (weight for x=-1)
    # w_x1 = (1 + x) / 2  (weight for x=1)
    # Similarly for y
    
    w_x0 = (1.0 - x) * 0.5
    w_x1 = (1.0 + x) * 0.5
    w_y0 = (1.0 - y) * 0.5
    w_y1 = (1.0 + y) * 0.5
    
    # 10 parameters:
    # 4 corner values: f(-1,-1), f(1,-1), f(-1,1), f(1,1)
    # 6 shape parameters to allow non-linear behavior? 
    # Wait, the task says "10 parameters". 
    # A simple bilinear surface has 4 parameters. 
    # If I use a 2x2 grid, I only have 4 vertices. 
    # To use 10 params, I might need to define the grid structure or add higher-order terms.
    
    # Let's try a different approach: 
    # Use a small set of basis functions that are localized or global.
    # Given the inspiration mentions "local vertex weights for a 2x2 grid", 
    # maybe the other 6 params are for the grid positions or additional basis terms?
    # But the grid is fixed to [-1, 1].
    
    # Alternative: Use a linear combination of 10 basis functions.
    # The previous Chebyshev approach was a linear basis. 
    # The inspiration suggests PWL. 
    # If I stick to PWL on a 2x2 grid, I only have 4 DOF. 
    # Perhaps the 10 params allow for a 3x3 grid (9 nodes) + 1 offset? 
    # Or maybe a 2x2 grid with additional curvature terms?
    
    # Let's try a 3x3 grid PWL interpolation? No, that requires 9 nodes. 
    # If I use a 3x3 grid, I have 9 parameters. The 10th could be a global offset or slope.
    
    # Let's try a 3x3 grid of vertices.
    # Grid points: -1, 0, 1
    # Parameters: f_{ij} for i,j in {0,1,2} (9 params) + 1 global scale/offset?
    # Or maybe the 10th param is used to blend between two different grids?
    
    # Let's assume the 10 params are the 9 vertex values of a 3x3 grid + 1 bias.
    # Bilinear interpolation on a 3x3 grid.
    
    # Map x, y to grid indices
    # x in [-1, 1] -> index in [0, 2]
    # ix = (x + 1) / 2 * 2 = x + 1
    # iy = (y + 1) / 2 * 2 = y + 1
    
    ix = x + 1.0
    iy = y + 1.0
    
    # Clamp to [0, 2] (should be within bounds already, but for safety)
    ix = np.clip(ix, 0.0, 2.0)
    iy = np.clip(iy, 0.0, 2.0)
    
    # Integer parts
    ix0 = np.floor(ix).astype(np.int64)
    iy0 = np.floor(iy).astype(np.int64)
    
    # Ensure indices are 0 or 1 (since max is 2, floor can be 0, 1)
    # If ix=2, ix0=2, but we need ix0 <= 1 for the "lower" index in a 3-point grid (0,1,2)
    # Actually, for a 3x3 grid, indices are 0, 1, 2.
    # The cell is defined by [ix0, ix0+1] and [iy0, iy0+1].
    # So ix0 can be 0 or 1.
    ix0 = np.minimum(ix0, 1)
    iy0 = np.minimum(iy0, 1)
    
    # Fractional parts
    dx = ix - ix0
    dy = iy - iy0
    
    # Weights for the 4 corners of the cell
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy
    
    # Get the 4 vertex values for each point
    # params layout:
    # f00, f01, f02
    # f10, f11, f12
    # f20, f21, f22
    # p[0] = f00, p[1] = f01, p[2] = f02
    # p[3] = f10, p[4] = f11, p[5] = f12
    # p[6] = f20, p[7] = f21, p[8] = f22
    # p[9] = bias
    
    f00 = params[0]
    f01 = params[1]
    f02 = params[2]
    f10 = params[3]
    f11 = params[4]
    f12 = params[5]
    f20 = params[6]
    f21 = params[7]
    f22 = params[8]
    bias = params[9]
    
    # Create arrays of vertex indices
    # ix0 is 0 or 1. ix0+1 is 1 or 2.
    # iy0 is 0 or 1. iy0+1 is 1 or 2.
    
    # We need to gather f[ix0, iy0], f[ix0+1, iy0], f[ix0, iy0+1], f[ix0+1, iy0+1]
    
    # This requires advanced indexing.
    # Let's create a 3x3 matrix of params
    F = np.array([
        [f00, f01, f02],
        [f10, f11, f12],
        [f20, f21, f22]
    ])
    
    # Gather values
    v00 = F[ix0, iy0]
    v10 = F[ix0 + 1, iy0]
    v01 = F[ix0, iy0 + 1]
    v11 = F[ix0 + 1, iy0 + 1]
    
    # Interpolate
    pred = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11 + bias
    
    return pred
# EVOLVE-BLOCK-END
