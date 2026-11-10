import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Polynomial + Learnable Frequency Sines/Cosines + Learnable Width Gaussian.
    
    y(x) = c0 + c1*x + c2*x^2 + c3*x^3 
           + A1*sin(w1*x) 
           + A2*cos(w2*x) 
           + A3*exp(-w3*x^2)
           
    params[0:4] -> c0, c1, c2, c3
    params[4:6] -> A1, w1
    params[6:8] -> A2, w2
    params[8:10]-> A3, w3
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Polynomial part
    c0, c1, c2, c3 = p[0], p[1], p[2], p[3]
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    # Sine part
    A1, w1 = p[4], p[5]
    # Clip frequency to avoid numerical issues with high freq
    w1_clipped = np.clip(w1, -20, 20)
    sine_term = A1 * np.sin(w1_clipped * x)
    
    # Cosine part
    A2, w2 = p[6], p[7]
    w2_clipped = np.clip(w2, -20, 20)
    cos_term = A2 * np.cos(w2_clipped * x)
    
    # Gaussian part
    A3, w3 = p[8], p[9]
    # Width must be positive for decay. Use abs or clip to small positive.
    # exp(-w3 * x^2). If w3 is negative, it explodes.
    # We want a bump, so w3 > 0.
    w3_clipped = np.clip(w3, 0.01, 20)
    gauss_term = A3 * np.exp(-w3_clipped * x**2)
    
    result = poly + sine_term + cos_term + gauss_term
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
