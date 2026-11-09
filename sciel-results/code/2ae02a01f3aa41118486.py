import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Fourier series basis with fixed orthogonal frequencies
    # Domain is [-2, 2], so fundamental frequency is pi/2
    # Using harmonics 1 through 5 for sine and cosine
    # This gives 10 terms total: cos(k*pi*x/2) and sin(k*pi*x/2) for k=1..5
    
    # Scale x to [-pi, pi] for standard Fourier basis
    # Or use x directly with appropriate frequencies
    # Let's use theta = x * pi / 2 so that x in [-2,2] maps to theta in [-pi, pi]
    theta = x * np.pi / 2.0
    
    # Cosine terms: cos(k*theta) for k=1..5
    cos_terms = np.column_stack([np.cos(k * theta) for k in range(1, 6)])
    
    # Sine terms: sin(k*theta) for k=1..5
    sin_terms = np.column_stack([np.sin(k * theta) for k in range(1, 6)])
    
    # Combine: p[0..4] for cosines, p[5..9] for sines
    y_hat = cos_terms @ p[0:5] + sin_terms @ p[5:10]
    
    return y_hat
