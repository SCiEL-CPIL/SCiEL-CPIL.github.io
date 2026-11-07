import numpy as np
import time

# Deadline guard: 150s wall-clock limit, use 145s to be safe
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Non-linear hybrid model with learnable basis function parameters.
    
    Parameter allocation (10 total):
    - p0: weight for constant
    - p1: weight for linear (xs)
    - p2: weight for quadratic (xs^2)
    - p3: weight for cubic (xs^3)
    - p4: tanh unit 1 center
    - p5: tanh unit 1 slope
    - p6: tanh unit 2 center
    - p7: tanh unit 2 slope
    - p8: RBF 1 center (in original x scale)
    - p9: RBF 2 center (in original x scale)
    
    The tanh units and RBFs have learnable locations/slopes, making the model
    non-linear in params. BFGS can adapt these to capture non-polynomial features.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x to [-1, 1] for better numerical conditioning
        xs = x / 2.0
        
        # Polynomial basis (fixed structure, learnable weights)
        p0 = np.ones_like(xs)
        p1 = xs.copy()
        p2 = xs**2
        p3 = xs**3
        
        # Learnable tanh units: tanh(slope * (xs - center))
        c1 = params[4]
        s1 = np.clip(params[5], -5.0, 5.0)
        c2 = params[6]
        s2 = np.clip(params[7], -5.0, 5.0)
        
        t1 = np.tanh(s1 * (xs - c1))
        t2 = np.tanh(s2 * (xs - c2))
        
        # Learnable Gaussian RBFs: exp(-(x - center)^2)
        r1c = params[8]
        r2c = params[9]
        
        r1 = np.exp(-(x - r1c)**2)
        r2 = np.exp(-(x - r2c)**2)
        
        # Weights for nonlinear terms are embedded in the structure:
        # We use fixed weights of 1.0 for tanh and RBF outputs, 
        # but their positions/slopes are learned.
        # To give BFGS more control, let's scale nonlinear terms by a small factor
        # and rely on the polynomial weights for the main fit.
        
        # Actually, let's assign explicit weights from a subset of params
        # Revised: use all 10 params as follows:
        # p0-p3: polynomial weights
        # p4-p7: tanh centers and slopes
        # p8-p9: RBF centers
        # The nonlinear terms get implicit weight 1.0
        
        result = (params[0] * p0 + 
                  params[1] * p1 + 
                  params[2] * p2 + 
                  params[3] * p3 + 
                  t1 + t2 + r1 + r2)
        
        return result
        
    except Exception:
        return np.zeros_like(x)
