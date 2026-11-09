import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check for non-finite inputs
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.zeros_like(x)
        
    # Structural Change: Tensor-product Fourier Basis
    # Using sine and cosine terms with frequency 1 (period 2) aligned to [-1, 1] domain.
    # Basis functions:
    # 0: 1 (constant)
    # 1: cos(pi*x)
    # 2: sin(pi*x)
    # 3: cos(pi*y)
    # 4: sin(pi*y)
    # 5: cos(pi*x)*cos(pi*y)
    # 6: sin(pi*x)*cos(pi*y)
    # 7: cos(pi*x)*sin(pi*y)
    # 8: sin(pi*x)*sin(pi*y)
    # 9: cos(2*pi*x)*cos(2*pi*y)  (higher frequency interaction for capacity)
    
    # Precompute trigonometric terms
    cos_x = np.cos(np.pi * x)
    sin_x = np.sin(np.pi * x)
    cos_y = np.cos(np.pi * y)
    sin_y = np.sin(np.pi * y)
    
    # Interaction terms
    cos_x_cos_y = cos_x * cos_y
    sin_x_cos_y = sin_x * cos_y
    cos_x_sin_y = cos_x * sin_y
    sin_x_sin_y = sin_x * sin_y
    
    # Higher frequency term (frequency 2)
    cos_2x = np.cos(2.0 * np.pi * x)
    cos_2y = np.cos(2.0 * np.pi * y)
    cos_2x_cos_2y = cos_2x * cos_2y
    
    pred = (params[0] * 1.0 + 
            params[1] * cos_x + 
            params[2] * sin_x + 
            params[3] * cos_y + 
            params[4] * sin_y + 
            params[5] * cos_x_cos_y + 
            params[6] * sin_x_cos_y + 
            params[7] * cos_x_sin_y + 
            params[8] * sin_x_sin_y + 
            params[9] * cos_2x_cos_2y)
            
    return pred
