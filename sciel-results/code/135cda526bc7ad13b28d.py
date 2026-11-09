"""Symbolic regression with 1 input: Parametric Gaussian Mixture + Polynomial."""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Quadratic Polynomial + 2 Gaussian Bumps.
    
    y(x) = w0 + w1*x + w2*x^2 + A1*exp(-B1*(x-C1)^2) + A2*exp(-B2*(x-C2)^2)
    
    Parameters:
    0: w0 (poly const)
    1: w1 (poly linear)
    2: w2 (poly quadratic)
    3: A1 (Gauss1 amp)
    4: B1 (Gauss1 width)
    5: C1 (Gauss1 center)
    6: A2 (Gauss2 amp)
    7: B2 (Gauss2 width)
    8: C2 (Gauss2 center)
    9: reserved (set to 0 or used for cubic term if needed, but let's stick to 8 active + 1 cubic + 1 quartic? 
       Actually, let's use 4 poly terms to be safe on global shape: w0..w3)
       
    Revised Params:
    0: w0
    1: w1
    2: w2
    3: w3
    4: A1
    5: B1
    6: C1
    7: A2
    8: B2
    9: C2
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params are 1D and length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
        
    # Extract parameters
    w0, w1, w2, w3 = params[0], params[1], params[2], params[3]
    A1, B1, C1 = params[4], params[5], params[6]
    A2, B2, C2 = params[7], params[8], params[9]
    
    # Polynomial part
    pred = w0 + w1 * x + w2 * x**2 + w3 * x**3
    
    # Gaussian 1
    # Clip B to ensure positive width for stability
    B1_safe = np.maximum(B1, 1e-6)
    g1 = A1 * np.exp(-B1_safe * (x - C1)**2)
    
    # Gaussian 2
    B2_safe = np.maximum(B2, 1e-6)
    g2 = A2 * np.exp(-B2_safe * (x - C2)**2)
    
    pred += g1 + g2
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
