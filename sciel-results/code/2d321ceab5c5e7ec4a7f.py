import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure params has exactly 10 elements
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Normalize x to [-1, 1]
    x_norm = x / 2.0
    
    # Composite basis: polynomial (4), trigonometric (3), tanh (3)
    # This covers polynomial, periodic, and saturation features simultaneously
    
    # Polynomial part: 1, x, x^2, x^3 (4 params)
    poly = np.vander(x_norm, N=4, increasing=True)  # shape (n, 4)
    
    # Trigonometric part: sin(x), cos(x), sin(2x) (3 params)
    trig = np.column_stack([
        np.sin(x_norm),
        np.cos(x_norm),
        np.sin(2.0 * x_norm)
    ])  # shape (n, 3)
    
    # Tanh/saturation part: tanh(x), tanh(2x), tanh(0.5x) (3 params)
    tanh_part = np.column_stack([
        np.tanh(x_norm),
        np.tanh(2.0 * x_norm),
        np.tanh(0.5 * x_norm)
    ])  # shape (n, 3)
    
    # Concatenate all basis functions: total 10 columns
    basis = np.hstack([poly, trig, tanh_part])  # shape (n, 10)
    
    # Linear combination
    return basis @ p
