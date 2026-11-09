"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed Gaussian RBF basis.
    # We define 10 fixed centers distributed across the input domain [0.5, 2]^3.
    # The parameters are the amplitudes of these RBFs.
    # This transforms the problem into a linear regression problem (if we had more params) 
    # or a simple amplitude fitting problem, which BFGS handles very well because 
    # the landscape is convex with respect to the amplitudes (given fixed centers).
    
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Flatten for easier distance calculation
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    
    n_points = x_flat.size
    
    # Define 10 fixed centers.
    # We want them spread out. 
    # A simple grid: 2x2x2 = 8 centers, plus 2 more in the middle?
    # Or just 10 random but deterministic points? 
    # Let's use a deterministic set of points to ensure reproducibility and coverage.
    # Domain is [0.5, 2.0].
    # Let's pick centers at:
    # (0.5, 0.5, 0.5), (0.5, 0.5, 2.0), (0.5, 2.0, 0.5), (0.5, 2.0, 2.0)
    # (2.0, 0.5, 0.5), (2.0, 0.5, 2.0), (2.0, 2.0, 0.5), (2.0, 2.0, 2.0)
    # (1.25, 1.25, 1.25) - center
    # (1.25, 0.5, 1.25) - edge mid
    # This gives good coverage of corners and center.
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 0.5, 1.25]
    ])
    
    # Standard deviation for the Gaussians.
    # Needs to be wide enough to cover the domain but narrow enough to localize.
    # Range is 1.5. A sigma of 0.5-0.8 seems reasonable.
    # If sigma is too small, the basis is too local and might not capture global trends well with only 10 centers.
    # If sigma is too large, it becomes like a polynomial.
    # Let's try sigma = 0.6.
    sigma = 0.6
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Calculate distances from each point to each center
    # x_flat: (N,), centers: (10, 3)
    # We want dist_sq: (N, 10)
    
    # Vectorized distance calculation
    # dx = x_flat[:, None] - centers[None, :, 0]
    # dy = y_flat[:, None] - centers[None, :, 1]
    # dz = z_flat[:, None] - centers[None, :, 2]
    # dist_sq = dx*dx + dy*dy + dz*dz
    
    # To be efficient and memory safe for N=200, this is fine.
    dx = x_flat[:, np.newaxis] - centers[np.newaxis, :, 0]
    dy = y_flat[:, np.newaxis] - centers[np.newaxis, :, 1]
    dz = z_flat[:, np.newaxis] - centers[np.newaxis, :, 2]
    
    dist_sq = dx*dx + dy*dy + dz*dz
    
    # Compute RBF values: exp(-dist_sq / (2*sigma^2))
    rbf_values = np.exp(-dist_sq / sigma_sq_2)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    
    # Prediction is rbf_values @ params
    # This is a linear combination of the RBFs.
    # This makes the optimization problem convex with respect to params,
    # which BFGS will solve very quickly and accurately.
    
    predictions = rbf_values @ params
    
    return predictions
# EVOLVE-BLOCK-END
