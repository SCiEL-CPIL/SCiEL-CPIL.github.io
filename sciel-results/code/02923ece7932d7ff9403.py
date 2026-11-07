import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a parametric sinusoidal model.
    
    Model: p0*sin(p1*x + p2*y + p3) + p4*cos(p5*x + p6*y + p7) + p8*x + p9*y
    
    This allows BFGS to optimize frequencies, phases, and amplitudes jointly,
    enabling exact recovery of functions like sin(3x+4y) which fixed-basis
    linear models cannot represent exactly.
    """
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n = len(x)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        # Pad with zeros if necessary, though evaluator should provide 10
        params = np.concatenate([params, np.zeros(10 - len(params))])
    
    # Extract parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    p4, p5, p6, p7 = params[4], params[5], params[6], params[7]
    p8, p9 = params[8], params[9]
    
    # Compute prediction
    # Handle potential numerical issues with large arguments to sin/cos
    # x, y are in [-1, 1], so p1*x + p2*y is bounded by |p1|+|p2|
    # If BFGS pushes frequencies very high, sin/cos are still stable but oscillatory
    arg1 = p1 * x + p2 * y + p3
    arg2 = p5 * x + p6 * y + p7
    
    pred = p0 * np.sin(arg1) + p4 * np.cos(arg2) + p8 * x + p9 * y
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Return zeros or something finite to avoid crash, though this shouldn't happen with sin/cos
        # unless params are inf/nan. BFGS usually keeps them finite.
        return np.zeros(n)
        
    return pred
