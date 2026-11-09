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
    Strategy: Local Piecewise Linear (PWL) interpolation on a 3x3 grid of vertices,
    plus a global quadratic correction term.
    
    The 9 parameters (params[0:9]) represent the values at the grid vertices:
    Grid points: x in {-1, 0, 1}, y in {-1, 0, 1}.
    Indices: 
    (0,0) -> x=-1, y=-1
    (1,0) -> x=0,  y=-1
    (2,0) -> x=1,  y=-1
    (0,1) -> x=-1, y=0
    (1,1) -> x=0,  y=0
    (2,1) -> x=1,  y=0
    (0,2) -> x=-1, y=1
    (1,2) -> x=0,  y=1
    (2,2) -> x=1,  y=1
    
    The 10th parameter (params[9]) scales a global term (x^2 + y^2) to capture
    curvature that the linear patches cannot.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract vertex values
    # Order: row-major (y then x) or column-major? Let's use index = i + 3*j where i is x-index, j is y-index
    # i=0 -> x=-1, i=1 -> x=0, i=2 -> x=1
    # j=0 -> y=-1, j=1 -> y=0, j=2 -> y=1
    v00 = params[0]  # x=-1, y=-1
    v10 = params[1]  # x=0,  y=-1
    v20 = params[2]  # x=1,  y=-1
    v01 = params[3]  # x=-1, y=0
    v11 = params[4]  # x=0,  y=0
    v21 = params[5]  # x=1,  y=0
    v02 = params[6]  # x=-1, y=1
    v12 = params[7]  # x=0,  y=1
    v22 = params[8]  # x=1,  y=1
    c_quad = params[9]
    
    # Map x, y to grid coordinates [0, 2]
    # x in [-1, 1] -> u in [0, 2]
    u = x + 1.0
    v = y + 1.0
    
    # Determine cell indices
    # Clip to [0, 1] to handle boundaries safely (floor of 2.0 is 2, but we need max 1 for cell start)
    i = np.clip(np.floor(u), 0, 1).astype(np.int64)
    j = np.clip(np.floor(v), 0, 1).astype(np.int64)
    
    # Local coordinates within the cell
    dx = u - i
    dy = v - j
    
    # Bilinear weights
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    
    # Gather vertex values for the 4 corners of each point's cell
    # We need to index into the 3x3 grid of vertices
    # Create a 3x3 array of vertex values
    verts = np.array([
        [v00, v10, v20],
        [v01, v11, v21],
        [v02, v12, v22]
    ])
    
    # Flatten for indexing or use advanced indexing
    # i, j are the bottom-left corner indices
    # Top-left: (i, j+1)
    # Bottom-right: (i+1, j)
    # Top-right: (i+1, j+1)
    
    val00 = verts[i, j]
    val10 = verts[i + 1, j]
    val01 = verts[i, j + 1]
    val11 = verts[i + 1, j + 1]
    
    # Interpolate
    pwl_val = w00 * val00 + w10 * val10 + w01 * val01 + w11 * val11
    
    # Add global quadratic correction
    quad_term = c_quad * (x**2 + y**2)
    
    return pwl_val + quad_term

# EVOLVE-BLOCK-END
