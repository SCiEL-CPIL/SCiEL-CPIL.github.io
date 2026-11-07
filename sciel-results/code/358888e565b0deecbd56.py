import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Softplus basis: sum_i w_i * softplus(s_i * (x - c_i))
    # 10 terms with fixed centers and slopes, only weights are free
    
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Slopes: symmetric pattern to capture both increasing and decreasing features
    # Alternate positive and negative slopes for flexibility
    slopes = np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
    
    # Compute softplus activations: log(1 + exp(z))
    z = slopes[np.newaxis, :] * (x[:, np.newaxis] - centers[np.newaxis, :])
    
    # Numerically stable softplus
    softplus = np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -50, 50))))
    
    # Linear combination with weights
    w = params
    
    return softplus @ w
