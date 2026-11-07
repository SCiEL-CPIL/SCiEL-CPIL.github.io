import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped driven Duffing oscillator model.
    
    Model: a = x''
    x'' + c1*x' + c2*x + c3*x^2 + c4*x^3 = A*sin(w*t + phi) + offset
    
    So: a = -c1*v - c2*x - c3*x^2 - c4*x^3 + A*sin(w*t + phi) + offset
    
    Parameters (10 total):
    p0: offset (static bias)
    p1: c1 (linear damping coefficient)
    p2: c2 (linear stiffness)
    p3: c3 (quadratic stiffness)
    p4: c4 (cubic stiffness)
    p5: A (drive amplitude)
    p6: w (drive frequency)
    p7: phi (drive phase)
    p8: scale_x (scaling for x to improve conditioning)
    p9: scale_v (scaling for v to improve conditioning)
    
    The model uses scaled variables: X = x/p8, V = v/p9
    a = p0 - p1*V - p2*X - p3*X^2 - p4*X^3 + p5*sin(p6*t + p7)
    """
    start_time = time.time()
    
    # Extract parameters
    offset = params[0]
    c1 = params[1]  # damping
    c2 = params[2]  # linear stiffness
    c3 = params[3]  # quadratic stiffness
    c4 = params[4]  # cubic stiffness
    A = params[5]   # drive amplitude
    w = params[6]   # drive frequency
    phi = params[7] # drive phase
    scale_x = params[8]
    scale_v = params[9]
    
    # Avoid division by zero
    eps = 1e-6
    sx = scale_x if abs(scale_x) > eps else eps
    sv = scale_v if abs(scale_v) > eps else eps
    
    # Scale inputs
    X = x / sx
    V = v / sv
    
    # Time safety check
    if time.time() - start_time > 130:
        # Return simplified prediction
        pred = offset - c1 * V - c2 * X
        return np.where(np.isfinite(pred), pred, 0.0)
    
    # Compute Duffing terms
    X2 = X * X
    X3 = X2 * X
    
    # Drive term
    drive = A * np.sin(w * t + phi)
    
    # Combine
    pred = offset - c1 * V - c2 * X - c3 * X2 - c4 * X3 + drive
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to linear terms only
        pred = offset - c1 * V - c2 * X
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred

# EVOLVE-BLOCK-END
