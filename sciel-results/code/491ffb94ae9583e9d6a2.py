import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Strategy: Sum of Gaussian Basis Functions (Radial Basis Functions)
    # Model: y(x) = sum_{i=1}^{5} A_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))
    # 
    # Parameters (10 total):
    # p[0:5] -> Amplitudes (A_i)
    # p[5:10] -> Means (mu_i)
    #
    # Note: The widths (sigma_i) are fixed to a reasonable value (0.5) to maintain
    # linearity in the parameters, which helps BFGS converge more reliably from
    # the all-ones start point. If the function has sharp features, the fixed
    # width might be too wide or narrow, but the adaptive placement of means
    # and amplitudes provides significant flexibility compared to global polynomials.
    # 
    # Why this might work:
    # 1. Localized basis: Can approximate non-smooth or multi-modal functions better
    #    than global polynomials (which suffer from Runge's phenomenon) or rational
    #    functions (which are prone to poles and optimization instability).
    # 2. Linear in parameters: The objective function is convex with respect to A_i
    #    and mu_i is non-linear, but the loss landscape is generally smoother than
    #    rational functions.
    # 3. Stability: Gaussians are bounded and smooth, avoiding division by zero issues.
    
    A = p[0:5]
    mu = p[5:10]
    
    # Fixed widths for stability
    sigma = 0.5
    
    # Compute the sum of Gaussians
    # y = sum A_i * exp(-((x - mu_i)^2) / (2 * sigma^2))
    
    y = np.zeros_like(x)
    
    for i in range(5):
        # Calculate distance from mean
        diff = x - mu[i]
        # Calculate exponent
        exponent = -(diff * diff) / (2.0 * sigma * sigma)
        # Calculate Gaussian
        g = np.exp(exponent)
        # Add weighted contribution
        y += A[i] * g
        
    # Safety clip to prevent overflow in MSE calculation, though Gaussians are bounded
    # if A is bounded. However, BFGS might push A to large values if it fails to converge.
    # Clip to a large but safe value.
    y = np.clip(y, -1e100, 1e100)
    
    return y
