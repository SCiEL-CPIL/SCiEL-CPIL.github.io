import numpy as np
from scipy.optimize import minimize

def construct():
    # Coefficients for H0, H4, H8
    # H12 is derived
    
    # Weights for I0 and I2
    # I0 terms: c0^2 * 1, c4^2 * 384, c8^2 * 10321920, c12^2 * 1961988556800
    # I2 terms: c4^2 * 1536, c8^2 * 82575360, c12^2 * 23543862681600
    
    w0_0 = 1.0
    w0_4 = 384.0
    w0_8 = 10321920.0
    w0_12 = 1961988556800.0
    
    w2_4 = 1536.0
    w2_8 = 82575360.0
    w2_12 = 23543862681600.0
    
    # Constraint: c0 + 12 c4 + 1680 c8 + 479040 c12 = 0
    # c0 = -12 c4 - 1680 c8 - 479040 c12
    
    def objective(params):
        c4, c8, c12 = params
        c0 = -12 * c4 - 1680 * c8 - 479040 * c12
        
        # Scale to prevent overflow
        # The terms are huge. Let's divide by a large constant or use logs?
        # Actually, numpy can handle floats up to 1e308.
        # 1961988556800 is ~ 2e12. Squared is 4e24.
        # If c12 is 1, term is 2e24. Safe.
        
        I0 = w0_0 * c0**2 + w0_4 * c4**2 + w0_8 * c8**2 + w0_12 * c12**2
        I2 = w2_4 * c4**2 + w2_8 * c8**2 + w2_12 * c12**2
        
        if I0 == 0:
            return 1e10
        return I2 / I0

    # Initial guess
    # The seed was [-1.13, 0.04, 0.000308]
    # Let's try near the seed or random starts.
    
    best_val = np.inf
    best_params = None
    
    # Try a few starting points
    starts = [
        [0.04, 0.000308, 0.0], # Based on seed c4, c8, c12=0? No, seed c12 is derived.
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0]
    ]
    
    # Normalize starts to unit norm to be consistent
    for s in starts:
        s = np.array(s)
        norm = np.linalg.norm(s)
        if norm > 0:
            s = s / norm
        
        res = minimize(objective, s, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    # Refine with BFGS if possible (need gradient)
    # Gradient of I2/I0 = (I2' I0 - I2 I0') / I0^2
    # But Nelder-Mead is robust enough for 3 vars.
    
    if best_params is None:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
        
    c4, c8, c12 = best_params
    c0 = -12 * c4 - 1680 * c8 - 479040 * c12
    
    return np.array([c0, c4, c8])
