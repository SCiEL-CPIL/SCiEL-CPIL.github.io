"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 145.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Piecewise-Constant (PWL) Partition.
    
    The current linear basis approach fails if the target function has non-linear 
    features that cannot be well-approximated by a small set of global basis functions 
    (e.g., sharp transitions, complex interactions). 
    
    This strategy partitions the input domain [-1, 1] x [-1, 1] into a grid of tiles.
    We use 10 parameters to represent the output value in each of 10 specific regions.
    
    To maximize flexibility with only 10 parameters, we use a non-uniform grid or 
    a specific tiling that covers the domain. A 2x5 grid (2 rows, 5 columns) or 
    5x2 grid gives 10 tiles. Alternatively, a 3x3 grid has 9 tiles, leaving 1 for 
    a global offset or a specific feature. 
    
    Let's use a 2x5 grid:
    - x ranges from -1 to 1. Split into 5 intervals: [-1, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.2, 0.6], [0.6, 1]
    - y ranges from -1 to 1. Split into 2 intervals: [-1, 0], [0, 1]
    
    Each parameter p[i] corresponds to the predicted value in tile i.
    This allows the model to fit any function that is approximately constant 
    within each tile. Since BFGS optimizes the parameters, it will find the 
    optimal constant value for each region (essentially performing a local 
    averaging/minimization within each tile).
    
    This is structurally different from a global basis because it allows 
    discontinuities and local features that global smooth bases (poly/trig) 
    struggle to capture with few parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params is length 10
    if params.shape[0] != 10:
        # Fallback if shape is wrong, though evaluator guarantees 10
        return np.zeros_like(x)
        
    # Define grid boundaries
    # x boundaries: 5 intervals
    x_bounds = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    # y boundaries: 2 intervals
    y_bounds = np.array([-1.0, 0.0, 1.0])
    
    # Determine which tile each point falls into
    # np.digitize returns the index of the bin to which each value of the input array belongs.
    # x_idx will be in [0, 4] for 5 bins
    # y_idx will be in [0, 1] for 2 bins
    
    # Note: np.digitize(x, x_bounds) gives 1..5 for values in [x_bounds[0], x_bounds[-1]]
    # We want 0..4 and 0..1
    x_idx = np.digitize(x, x_bounds) - 1
    y_idx = np.digitize(y, y_bounds) - 1
    
    # Clip indices to valid range to handle edge cases (e.g., x=1.0 exactly)
    x_idx = np.clip(x_idx, 0, 4)
    y_idx = np.clip(y_idx, 0, 1)
    
    # Map (x_idx, y_idx) to a single index 0..9
    # Layout:
    # y=1 (top row):   indices 0, 1, 2, 3, 4
    # y=0 (bottom row): indices 5, 6, 7, 8, 9
    # So index = y_idx * 5 + x_idx
    
    tile_idx = y_idx * 5 + x_idx
    
    # Gather the parameter for each tile
    # params[tile_idx] will give the prediction for each point
    pred = params[tile_idx]
    
    return pred

# EVOLVE-BLOCK-END
