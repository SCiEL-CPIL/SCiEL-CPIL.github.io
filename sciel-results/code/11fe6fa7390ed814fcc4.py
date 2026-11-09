import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable Fourier-Linear model.
    
    Structure:
    pred = c_x*x + c_y*y + c_z*z 
         + a_x*sin(w*x) + b_x*cos(w*x)
         + a_y*sin(w*y) + b_y*cos(w*y)
         + a_z*sin(w*z) + b_z*cos(w*z)
         
    Parameters (10 total):
    p[0]: c_x (linear coeff for x)
    p[1]: c_y (linear coeff for y)
    p[2]: c_z (linear coeff for z)
    p[3]: a_x (sine amp for x)
    p[4]: b_x (cos amp for x)
    p[5]: a_y (sine amp for y)
    p[6]: b_y (cos amp for y)
    p[7]: a_z (sine amp for z)
    p[8]: b_z (cos amp for z)
    p[9]: w (shared frequency)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    c_x, c_y, c_z = p[0], p[1], p[2]
    a_x, b_x = p[3], p[4]
    a_y, b_y = p[5], p[6]
    a_z, b_z = p[7], p[8]
    w = p[9]
    
    # Calculate trigonometric terms
    # To avoid overflow or instability with large w, we could clamp, 
    # but BFGS usually keeps params reasonable. 
    # Inputs are in [0.5, 2], so w*x is in [0.5w, 2w].
    sin_x = np.sin(w * x)
    cos_x = np.cos(w * x)
    sin_y = np.sin(w * y)
    cos_y = np.cos(w * y)
    sin_z = np.sin(w * z)
    cos_z = np.cos(w * z)
    
    # Combine
    pred = (c_x * x + c_y * y + c_z * z +
            a_x * sin_x + b_x * cos_x +
            a_y * sin_y + b_y * cos_y +
            a_z * sin_z + b_z * cos_z)
            
    return pred
