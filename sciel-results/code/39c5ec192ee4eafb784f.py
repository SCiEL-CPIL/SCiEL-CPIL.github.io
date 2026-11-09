"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration using
    local linearization with Gaussian RBFs in the (x, v) phase plane.
    
    y = sum_k G_k(x, v) * (a_k + b_k*x + c_k*v) + d*sin(t) + e*cos(t) + f*x*v + g*x^2
    
    Where G_k are Gaussian basis functions centered at (cx_k, cv_k).
    This allows the local spring (b_k) and damping (c_k) coefficients to vary
    across the phase plane, capturing non-additive nonlinearities.
    """
    # Fixed RBF centers and width
    # Training ranges: x in [-0.34, 0.36], v in [-0.21, 0.17]
    cx0, cv0 = -0.1, -0.1
    cx1, cv1 = 0.1, 0.1
    sigma = 0.25
    
    # Compute Gaussian bases
    dx0 = x - cx0
    dv0 = v - cv0
    r2_0 = dx0**2 + dv0**2
    
    dx1 = x - cx1
    dv1 = v - cv1
    r2_1 = dx1**2 + dv1**2
    
    G0 = np.exp(-r2_0 / (2 * sigma**2))
    G1 = np.exp(-r2_1 / (2 * sigma**2))
    
    # Local linear terms multiplied by RBFs
    # Params 0-2: for G0 (const, x, v)
    # Params 3-5: for G1 (const, x, v)
    local_0 = params[0] * G0 + params[1] * x * G0 + params[2] * v * G0
    local_1 = params[3] * G1 + params[4] * x * G1 + params[5] * v * G1
    
    # Global terms
    # Params 6-7: driving force
    # Params 8-9: global nonlinear couplings
    drive = params[6] * np.sin(t) + params[7] * np.cos(t)
    global_nl = params[8] * x * v + params[9] * x * x
    
    return local_0 + local_1 + drive + global_nl

# EVOLVE-BLOCK-END
