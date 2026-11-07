import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based model for E. coli growth rate.
    
    Uses 10 Gaussian radial basis functions with fixed centers in the normalized 
    input space. The parameters are the weights for these basis functions.
    This allows for flexible, non-polynomial response surfaces that can capture 
    local maxima, plateaus, and complex interactions better than a global quadratic.
    
    Inputs:
    b: Population density [1, 5]
    s: Substrate concentration [1, 5]
    temp: Temperature [1, 40]
    pH: pH level [2, 12]
    params: Array of 10 weights
    """
    # 1. Scale inputs to [-1, 1]
    # b: [1, 5] -> center 3, range 4
    x_b = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, range 4
    x_s = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, range 39
    x_t = (temp - 20.5) / 19.5
    # pH: [2, 12] -> center 7, range 10
    x_p = (pH - 7.0) / 5.0

    # 2. Define 10 fixed centers in the [-1, 1]^4 space
    # We use the 4 corners, 4 face-centers (midpoints of axes), and the global center, 
    # plus one more to ensure full rank/coverage. 
    # Actually, let's use a symmetric set:
    # 4 Corners: (1,1,1,1), (1,1,-1,-1), (1,-1,1,-1), (-1,1,1,-1) -- wait, this is just 4.
    # Let's use all 4 "positive" corners and 4 "negative" corners? No, that's 8.
    # Let's use:
    # 1. (1, 1, 1, 1)
    # 2. (-1, -1, -1, -1)
    # 3. (1, -1, 1, -1)
    # 4. (-1, 1, -1, 1)
    # 5. (1, 1, -1, -1)
    # 6. (-1, -1, 1, 1)
    # 7. (1, -1, -1, 1)
    # 8. (-1, 1, 1, -1)
    # 9. (0, 0, 0, 0)
    # 10. (1, 0, 0, 0) ? Or maybe (0, 1, 0, 0)?
    # To be safe and symmetric, let's add the other 4 corners? 
    # Total corners are 16. We only have 10 params.
    # A good spread is:
    # 4 corners with all signs same: (1,1,1,1), (-1,-1,-1,-1) ... wait, that's 2.
    # Let's just pick 10 well-distributed points.
    
    centers = np.array([
        [ 1.0,  1.0,  1.0,  1.0],  # 0
        [-1.0, -1.0, -1.0, -1.0],  # 1
        [ 1.0, -1.0,  1.0, -1.0],  # 2
        [-1.0,  1.0, -1.0,  1.0],  # 3
        [ 1.0,  1.0, -1.0, -1.0],  # 4
        [-1.0, -1.0,  1.0,  1.0],  # 5
        [ 1.0, -1.0, -1.0,  1.0],  # 6
        [-1.0,  1.0,  1.0, -1.0],  # 7
        [ 0.0,  0.0,  0.0,  0.0],  # 8 (Center)
        [ 1.0,  0.0,  0.0,  0.0]   # 9 (Extra directional center, e.g., high density effect)
    ])
    
    # Gaussian width. 
    # Distance in 4D space between corners is up to 4.
    # Sigma = 1.5 provides good overlap.
    sigma = 1.5
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # 3. Compute distances and Gaussian kernels
    # Input shape: (N, 4)
    # Centers shape: (10, 4)
    
    # Efficient distance calculation
    # diff = x[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist_sq = np.sum(diff**2, axis=2)
    
    # Alternative vectorized approach without huge intermediate array if N is large, 
    # but N=7500 is small enough for standard broadcasting.
    x = np.stack([x_b, x_s, x_t, x_p], axis=-1) # Shape (N, 4)
    
    # (N, 1, 4) - (1, 10, 4) -> (N, 10, 4)
    diff = x[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2) # Shape (N, 10)
    
    # Gaussian values
    # K = exp(-dist_sq * inv_2_sigma2)
    K = np.exp(-dist_sq * inv_2_sigma2) # Shape (N, 10)
    
    # 4. Linear combination
    # params shape (10,)
    # y = K @ params
    growth = K @ params
    
    return growth
