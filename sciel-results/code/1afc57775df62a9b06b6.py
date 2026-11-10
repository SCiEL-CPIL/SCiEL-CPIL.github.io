"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a small neural network structure
    with 3 tanh activation units. This allows for non-linear, non-polynomial
    shapes (e.g., saturation, S-curves) which global polynomials may struggle
    to approximate efficiently.
    
    Model: y = p0 + p1*tanh(p2*x + p3) + p4*tanh(p5*x + p6) + p7*tanh(p8*x + p9)
    """
    start_time = time.time()
    x = np.asarray(x, dtype=float)
    
    # Safety margin: 10 seconds before the 120s limit
    DEADLINE = 110.0
    
    # Extract parameters
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        # Pad with zeros if necessary, though evaluator should provide 10
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p

    # Calculate activations
    # tanh is stable for all real inputs
    a1 = np.tanh(p2 * x + p3)
    a2 = np.tanh(p5 * x + p6)
    a3 = np.tanh(p8 * x + p9)
    
    # Combine
    y = p0 + p1 * a1 + p4 * a2 + p7 * a3
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        # Return a large constant to signal failure to the optimizer, 
        # but avoid NaNs which crash BFGS. 
        # The contract says non-finite error costs 1e30 during fitting.
        # Returning inf/NaN might be handled by the evaluator, but returning 
        # a large finite number is safer for the gradient step if it happens.
        # However, the contract says "non-finite test predictions make the result invalid".
        # So we should try to avoid this. 
        # If params lead to inf, it's likely due to overflow in exp inside tanh? 
        # No, tanh is bounded. 
        # Overflow can happen in p2*x if p2 is huge. 
        # If p2 is 1e300, p2*x is inf. tanh(inf) is 1. 
        # So y should be finite unless p1 is inf.
        # If params are inf, the fit is already broken.
        pass

    return y
# EVOLVE-BLOCK-END
