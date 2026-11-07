"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Fixed centers for RBF network, distributed across the input domain [-2, 2]
    # We use 5 centers to utilize all 10 parameters (5 weights + 5 biases? No, just 5 weights + 5 centers? 
    # Wait, params are fixed length 10. 
    # If we fix centers, we have 10 free weights.
    # Let's use 10 RBFs with fixed centers to maximize flexibility within the convex constraint.
    
    centers = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    # Actually, domain is [-2, 2]. Let's space them evenly in [-2, 2].
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width (sigma) for Gaussian kernels
    # A width of 0.5 allows local approximation while maintaining smoothness
    sigma = 0.5
    
    # Compute the RBF basis matrix
    # X is (N, 1), Centers is (10,)
    # Diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (x - c)^2 / sigma^2)
    # To avoid underflow/overflow issues and keep values reasonable:
    # exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # params is a vector of length 10.
    # The model is: y = sum(w_i * phi_i(x))
    # This is linear in params.
    # prediction = basis @ params
    
    # Ensure params is a 1D array
    p = params.ravel()
    
    pred = basis @ p
    
    return pred
# EVOLVE-BLOCK-END
