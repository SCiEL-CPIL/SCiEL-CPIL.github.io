import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract centers and amplitudes
    # params[0:5] -> centers
    # params[5:10] -> amplitudes
    
    # Map centers to [-2, 2] using tanh
    # This ensures centers stay within the input domain, 
    # preventing optimization from drifting to regions where 
    # gradients vanish (if centers went to infinity)
    raw_centers = params[:5]
    centers = 2.0 * np.tanh(raw_centers)
    
    amplitudes = params[5:10]
    
    # Gaussian RBF with fixed width
    sigma = 1.0
    inv_2_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    pred = np.zeros_like(x)
    
    for i in range(5):
        diff = x - centers[i]
        # exp(-diff^2 / (2 sigma^2))
        pred += amplitudes[i] * np.exp(-diff * diff * inv_2_sigma_sq)
        
    return pred
