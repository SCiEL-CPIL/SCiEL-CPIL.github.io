"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Dense Fourier-RBF hybrid basis for nonlinear oscillator dynamics.
    
    Uses 4 RBF centers in (x,v) space combined with Fourier time terms,
    providing localized basis functions that can capture fine structure
    in the oscillator's phase-space dynamics.
    
    Basis terms (10 total):
    1-4: RBF(x,v; center_i) * sin(t) for i=0..3
    5-8: RBF(x,v; center_i) * cos(t) for i=0..3
    9:   RBF(x,v; center_0)  (time-independent local response)
    10:  RBF(x,v; center_1)  (time-independent local response)
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Scale inputs to roughly [-1, 1] range
    scale_x = 0.35
    scale_v = 0.20
    
    xs = x / scale_x
    vs = v / scale_v
    
    # Define 4 RBF centers with good spatial coverage
    # Center 0: near origin
    cx0, cv0 = 0.0, 0.0
    # Center 1: positive x, negative v (quadrant 4)
    cx1, cv1 = 0.6, -0.5
    # Center 2: negative x, positive v (quadrant 2)
    cx2, cv2 = -0.6, 0.5
    # Center 3: positive x, positive v (quadrant 1)
    cx3, cv3 = 0.5, 0.4
    
    # RBF width - smaller for more localized response
    width_sq = 4.0  # dividing by 4 instead of 3 for smaller widths
    
    # Compute squared distances to each center
    dx0 = xs - cx0
    dv0 = vs - cv0
    r2_0 = dx0 * dx0 + dv0 * dv0
    
    dx1 = xs - cx1
    dv1 = vs - cv1
    r2_1 = dx1 * dx1 + dv1 * dv1
    
    dx2 = xs - cx2
    dv2 = vs - cv2
    r2_2 = dx2 * dx2 + dv2 * dv2
    
    dx3 = xs - cx3
    dv3 = vs - cv3
    r2_3 = dx3 * dx3 + dv3 * dv3
    
    # Gaussian RBFs
    rbf0 = np.exp(-r2_0 / width_sq)
    rbf1 = np.exp(-r2_1 / width_sq)
    rbf2 = np.exp(-r2_2 / width_sq)
    rbf3 = np.exp(-r2_3 / width_sq)
    
    # Time Fourier terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Combine: RBF * Fourier + RBF (time-independent)
    result = (p0 * rbf0 * sin_t +
              p1 * rbf1 * sin_t +
              p2 * rbf2 * sin_t +
              p3 * rbf3 * sin_t +
              p4 * rbf0 * cos_t +
              p5 * rbf1 * cos_t +
              p6 * rbf2 * cos_t +
              p7 * rbf3 * cos_t +
              p8 * rbf0 +
              p9 * rbf1)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
    
    # Time guard
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END
