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
    
    Instead of a global polynomial basis which assumes smoothness and specific functional forms,
    we partition the domain into 4 quadrants (cells) using the axes x=0 and y=0.
    In each cell, we approximate the function as a linear function of (x, y).
    
    The 10 parameters are the values of the function at the 10 unique vertices of this grid:
    Grid points:
    (-1, -1), (0, -1), (1, -1)
    (-1,  0), (0,  0), (1,  0)
    (-1,  1), (0,  1), (1,  1)
    
    This results in 9 vertices? No, 3x3 grid = 9 vertices.
    Wait, I have 10 parameters.
    Let's use a 3x3 grid of vertices. That's 9 parameters.
    What is the 10th?
    Maybe a global constant offset? Or maybe the grid is not uniform?
    
    Let's stick to the 3x3 grid (9 vertices) and use the 10th parameter as a global bias or 
    perhaps adjust the grid to be 3x4? No, let's keep it simple.
    
    Actually, a 3x3 grid of bilinear interpolation uses 9 parameters (the vertex values).
    To use 10, we can add a global constant term that shifts all vertices, or 
    we can make the grid slightly non-uniform?
    
    Better idea: Use a 3x3 grid of vertices (9 params) and the 10th parameter as a global 
    multiplicative scale? No, that's non-linear in params.
    
    Let's use a 3x3 grid of vertices (9 params) and the 10th parameter as a constant 
    added to the entire prediction. This is equivalent to shifting all vertices by the same amount,
    but it decouples the "local" shape from the "global" offset, which might help optimization.
    
    Parameters:
    p[0..8]: Values at the 9 grid points (-1,-1), (0,-1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1)
    p[9]: Global constant offset
    
    The prediction for a point (x, y) is:
    1. Determine which cell it falls into.
    2. Compute local bilinear interpolation weights.
    3. Predict = sum(weights * vertex_values) + p[9]
    
    This is a linear function of params, so BFGS can fit it easily.
    It can represent any piecewise-bilinear function over this grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Grid vertices:
    # Row 0: y = -1, x in [-1, 0, 1]
    # Row 1: y =  0, x in [-1, 0, 1]
    # Row 2: y =  1, x in [-1, 0, 1]
    
    # We define the grid lines at x = -1, 0, 1 and y = -1, 0, 1.
    
    # Determine local coordinates in each cell.
    # For x:
    # If x < 0, it's in the left cell (between -1 and 0). Local x in [0, 1] is (x - (-1)) / (0 - (-1)) = x + 1.
    # If x >= 0, it's in the right cell (between 0 and 1). Local x in [0, 1] is (x - 0) / (1 - 0) = x.
    
    # For y:
    # If y < 0, it's in the bottom cell (between -1 and 0). Local y in [0, 1] is y + 1.
    # If y >= 0, it's in the top cell (between 0 and 1). Local y in [0, 1] is y.
    
    # We can vectorize this.
    # Let's define the weights for bilinear interpolation.
    
    # x_local: 0 to 1 within the cell
    x_left = x < 0
    x_right = ~x_left
    
    x_local = np.zeros_like(x)
    x_local[x_left] = x[x_left] + 1.0
    x_local[x_right] = x[x_right]
    
    # y_local: 0 to 1 within the cell
    y_bottom = y < 0
    y_top = ~y_bottom
    
    y_local = np.zeros_like(y)
    y_local[y_bottom] = y[y_bottom] + 1.0
    y_local[y_top] = y[y_top]
    
    # Bilinear weights:
    # w00: bottom-left of the cell (local 0,0)
    # w10: bottom-right of the cell (local 1,0)
    # w01: top-left of the cell (local 0,1)
    # w11: top-right of the cell (local 1,1)
    
    w00 = (1 - x_local) * (1 - y_local)
    w10 = x_local * (1 - y_local)
    w01 = (1 - x_local) * y_local
    w11 = x_local * y_local
    
    # Now map these local weights to the global vertex indices.
    # Global vertices are indexed as:
    # 0: (-1, -1), 1: (0, -1), 2: (1, -1)
    # 3: (-1,  0), 4: (0,  0), 5: (1,  0)
    # 6: (-1,  1), 7: (0,  1), 8: (1,  1)
    
    # For a point in the left-bottom cell (x<0, y<0):
    # Local (0,0) corresponds to global (-1, -1) -> index 0
    # Local (1,0) corresponds to global (0, -1)  -> index 1
    # Local (0,1) corresponds to global (-1,  0) -> index 3
    # Local (1,1) corresponds to global (0,  0)  -> index 4
    
    # For a point in the right-bottom cell (x>=0, y<0):
    # Local (0,0) corresponds to global (0, -1)  -> index 1
    # Local (1,0) corresponds to global (1, -1)  -> index 2
    # Local (0,1) corresponds to global (0,  0)  -> index 4
    # Local (1,1) corresponds to global (1,  0)  -> index 5
    
    # For a point in the left-top cell (x<0, y>=0):
    # Local (0,0) corresponds to global (-1, 0)  -> index 3
    # Local (1,0) corresponds to global (0,  0)  -> index 4
    # Local (0,1) corresponds to global (-1, 1)  -> index 6
    # Local (1,1) corresponds to global (0,  1)  -> index 7
    
    # For a point in the right-top cell (x>=0, y>=0):
    # Local (0,0) corresponds to global (0, 0)   -> index 4
    # Local (1,0) corresponds to global (1, 0)   -> index 5
    # Local (0,1) corresponds to global (0, 1)   -> index 7
    # Local (1,1) corresponds to global (1, 1)   -> index 8
    
    # We can compute the prediction by summing contributions from all 9 vertices.
    # Each vertex contributes its parameter value multiplied by the sum of weights from cells that use it.
    
    # Let's create a matrix of contributions.
    # For each point, we calculate the weight for each of the 9 global vertices.
    
    # Initialize weights array (N, 9)
    N = len(x)
    weights = np.zeros((N, 9), dtype=np.float64)
    
    # Left-Bottom (x<0, y<0):
    mask_lb = x_left & y_bottom
    if np.any(mask_lb):
        weights[mask_lb, 0] = w00[mask_lb]
        weights[mask_lb, 1] = w10[mask_lb]
        weights[mask_lb, 3] = w01[mask_lb]
        weights[mask_lb, 4] = w11[mask_lb]
        
    # Right-Bottom (x>=0, y<0):
    mask_rb = x_right & y_bottom
    if np.any(mask_rb):
        weights[mask_rb, 1] = w00[mask_rb]
        weights[mask_rb, 2] = w10[mask_rb]
        weights[mask_rb, 4] = w01[mask_rb]
        weights[mask_rb, 5] = w11[mask_rb]
        
    # Left-Top (x<0, y>=0):
    mask_lt = x_left & y_top
    if np.any(mask_lt):
        weights[mask_lt, 3] = w00[mask_lt]
        weights[mask_lt, 4] = w10[mask_lt]
        weights[mask_lt, 6] = w01[mask_lt]
        weights[mask_lt, 7] = w11[mask_lt]
        
    # Right-Top (x>=0, y>=0):
    mask_rt = x_right & y_top
    if np.any(mask_rt):
        weights[mask_rt, 4] = w00[mask_rt]
        weights[mask_rt, 5] = w10[mask_rt]
        weights[mask_rt, 7] = w01[mask_rt]
        weights[mask_rt, 8] = w11[mask_rt]
        
    # Prediction is weights @ params[0:9] + params[9]
    pred = weights @ params[:9] + params[9]
    
    return pred

# EVOLVE-BLOCK-END
