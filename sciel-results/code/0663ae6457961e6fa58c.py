import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense Radial Basis Function (RBF) network.
    
    Strategy:
    - Place 10 Gaussian centers evenly across the input domain [-2, 2].
    - Use a fixed, relatively narrow bandwidth (sigma=0.25) to allow 
      the model to approximate local features and discontinuities.
    - The 10 parameters are the weights for these 10 Gaussian basis functions.
    - This transforms the problem into a linear regression problem, which 
      BFGS can solve very efficiently and robustly, avoiding local minima 
      associated with non-linear basis terms.
    """
    # 10 centers evenly spaced in [-2, 2]
    # Centers: -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0? 
    # Wait, 10 points. 
    # np.linspace(-2, 2, 10) gives:
    # -2.0, -1.555, -1.111, -0.666, -0.222, 0.222, 0.666, 1.111, 1.555, 2.0
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Bandwidth. Smaller sigma means more localized. 
    # 0.25 is reasonable for 200 points over a range of 4.
    sigma = 0.25
    
    # Compute distances from each input x to each center
    # x shape: (N, 1) or (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = (x[i] - centers[j])^2
    
    x_flat = x.flatten()
    dist_sq = (x_flat[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # Compute Gaussian values: exp(-dist_sq / (2 * sigma^2))
    # sigma^2 = 0.0625
    # 2 * sigma^2 = 0.125
    gaussians = np.exp(-dist_sq / 0.125)
    
    # Linear combination with params
    # gaussians shape: (N, 10)
    # params shape: (10,)
    predictions = gaussians @ params
    
    # Reshape to match input shape if necessary, though (N,) is usually fine
    # The task says "return an array with one prediction per point"
    return predictions.reshape(x.shape)
