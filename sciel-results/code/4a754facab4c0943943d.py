"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression combining a low-degree Chebyshev polynomial base
    with 3 adaptive Gaussian bumps. The 10 parameters are split as:
      - 4 params: Chebyshev coefficients (T_0 to T_3) for the global trend
      - 3 pairs:  (center, width) for 3 Gaussian bumps
    
    This allows the BFGS optimizer to adaptively place and shape local features
    while maintaining a smooth global baseline.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.clip(np.asarray(params, dtype=np.float64), -1000.0, 1000.0)
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_3
    # T_0(u) = 1
    # T_1(u) = u
    # T_2(u) = 2u^2 - 1
    # T_3(u) = 4u^3 - 3u
    
    T0 = np.ones_like(u)
    T1 = u.copy()
    T2 = 2.0 * u * T1 - T0
    T3 = 2.0 * u * T2 - T1
    
    # Global trend from low-degree Chebyshev terms
    pred = params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3
    
    # Add 3 Gaussian bumps
    # Each bump: center = params[4+2*i], width = params[5+2*i]
    # Width is transformed to ensure positive value: width = 0.1 + abs(params[5+2*i])
    # Center is mapped from param to [-3, 3] range: center = 3.0 * tanh(params[4+2*i])
    
    for i in range(3):
        center_param = params[4 + 2 * i]
        width_param = params[5 + 2 * i]
        
        # Map center to [-3, 3] using tanh for smoothness and boundedness
        center = 3.0 * np.tanh(center_param)
        
        # Ensure positive width with a minimum to avoid division by zero
        width = 0.1 + np.abs(width_param)
        
        # Gaussian bump: exp(-((x - center)^2) / (2 * width^2))
        # Scale the bump height by a small factor to keep it manageable
        bump = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
        
        # Add the bump to the prediction with a fixed small amplitude
        # The amplitude is controlled by the width parameter's sign and magnitude
        # Use a soft scaling to keep it smooth
        amplitude = 0.5 * np.tanh(width_param)
        pred += amplitude * bump
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback: use only the Chebyshev terms
        pred = params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
    
    return pred
# EVOLVE-BLOCK-END
