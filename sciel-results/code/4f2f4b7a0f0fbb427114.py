"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    if len(x) == 0:
        return np.zeros(6)
    
    # 1. Local RBF Smoothing (Non-parametric step)
    # We create a set of "pseudo-targets" for a grid of points by locally averaging
    # the training data using Gaussian kernels. This denoises the data and captures
    # local structure better than a global polynomial might.
    
    # Determine a reasonable grid size based on data range and count
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(6)
    
    # Number of grid points for smoothing
    n_grid = min(200, max(50, len(x) // 10))
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    # Bandwidth selection: use a fraction of the range
    # Heuristic: bandwidth ~ range / sqrt(n) * constant
    bandwidth = (x_max - x_min) / np.sqrt(n_grid) * 0.5
    
    # Compute smoothed targets y_smooth on x_grid
    # y_smooth[i] = sum_j w_ij * y_j / sum_j w_ij
    # where w_ij = exp(-0.5 * ((x_grid[i] - x_j) / bandwidth)^2)
    
    # To save time, we do this in chunks if necessary, but n_grid is small.
    y_smooth = np.zeros(n_grid)
    for i in range(n_grid):
        if time.time() > deadline:
            break
        diff = x_grid[i] - x
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2)
        w_sum = np.sum(weights)
        if w_sum > 1e-10:
            y_smooth[i] = np.dot(weights, y) / w_sum
        else:
            # Fallback to global mean if no neighbors
            y_smooth[i] = np.mean(y)
    
    # Check if we finished the grid loop
    if time.time() > deadline:
        # If we ran out of time, fall back to a simple global fit on raw data
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return coeff
        except:
            return np.zeros(6)
    
    # 2. Global Polynomial Fit (Parametric step)
    # Fit a degree-5 polynomial to the smoothed data (x_grid, y_smooth)
    # This ensures the final output is a single shared polynomial as required.
    # We use least squares on the smoothed targets.
    
    # Construct Vandermonde matrix for degree 5
    V = np.vander(x_grid, N=6, increasing=True)
    
    # Solve least squares: V @ coeff = y_smooth
    try:
        coeff, residuals, rank, s = np.linalg.lstsq(V, y_smooth, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to polyfit
        try:
            coeff = np.polynomial.polynomial.polyfit(x_grid, y_smooth, 5)
        except:
            coeff = np.zeros(6)
    
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
    
    return coeff

# EVOLVE-BLOCK-END
