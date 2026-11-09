import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Non-linear parametric model with variable frequencies.
    
    The model is:
    z = c0 + c1*x + c2*y + c3*x*y 
        + A*sin(omega_x*x + phi_x) 
        + B*cos(omega_y*y + phi_y)
    
    Parameters:
    params[0]: c0 (bias)
    params[1]: c1 (x coeff)
    params[2]: c2 (y coeff)
    params[3]: c3 (xy coeff)
    params[4]: A (amplitude of sin term)
    params[5]: omega_x (frequency of x)
    params[6]: phi_x (phase of x)
    params[7]: B (amplitude of cos term)
    params[8]: omega_y (frequency of y)
    params[9]: phi_y (phase of y)
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    A = params[4]
    omega_x = params[5]
    phi_x = params[6]
    B = params[7]
    omega_y = params[8]
    phi_y = params[9]
    
    # Calculate polynomial part
    poly_part = c0 + c1 * x + c2 * y + c3 * x * y
    
    # Calculate oscillatory parts
    # Using np.sin and np.cos directly
    sin_part = A * np.sin(omega_x * x + phi_x)
    cos_part = B * np.cos(omega_y * y + phi_y)
    
    pred = poly_part + sin_part + cos_part
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Return a large constant to penalize invalid parameters
        return np.full_like(x, 1e10)
        
    return pred
