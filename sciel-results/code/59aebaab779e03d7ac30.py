"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a radial basis function (RBF) model with 5 fixed centers
    # The 5 parameters control the weights of the Gaussians
    # The remaining 5 parameters modulate the width (inverse variance) of each Gaussian
    # This gives a smooth, flexible non-polynomial basis that BFGS can optimize
    
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.array([-2.4, -1.2, 0.0, 1.2, 2.4])
    
    # Parameters: p[0:5] are weights, p[5:10] are inverse-width (precision) parameters
    weights = params[0:5]
    widths = params[5:10]
    
    # Ensure widths are positive and reasonable to avoid numerical issues
    # Use softplus-like transformation: width = 1 + exp(width_param) gives range (1, inf)
    # But we want widths to be tunable. Let's use: sigma = 1.0 / (1.0 + np.exp(width_param))
    # This gives sigma in (0, 1). Actually let's use a different scaling.
    # Let width_param control the inverse std: inv_std = 1.0 + 0.5 * np.tanh(width_param)
    # This gives inv_std in (0.5, 1.5), so std in (2/3, 2)
    
    inv_std = 1.0 + 0.5 * np.tanh(widths)
    
    # Compute the RBF values: for each center c_i, RBF_i(x) = exp(-0.5 * inv_std_i^2 * (x - c_i)^2)
    # This is a 5xN matrix where N is the number of points
    x_col = x.reshape(-1, 1)
    centers_row = centers.reshape(1, -1)
    
    # Compute squared distances: (x - c_i)^2 for all x and all i
    sq_dist = (x_col - centers_row) ** 2
    
    # Compute inv_std^2 for each center
    inv_std_sq = inv_std ** 2  # shape (5,)
    
    # RBF values: exp(-0.5 * inv_std_sq_i * sq_dist_ni)
    rbf_values = np.exp(-0.5 * inv_std_sq.reshape(1, -1) * sq_dist)
    
    # Linear combination with weights
    y = rbf_values @ weights
    
    return y
# EVOLVE-BLOCK-END
