"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral guess if we are too late, though evaluation should be fast
        return np.zeros_like(x, dtype=np.float64)

    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params
    
    # Normalize inputs to [-1, 1] range
    # Domain is [0.5, 2.0]
    # Center = 1.25, Half-Range = 0.75
    x_c = (x - 1.25) / 0.75
    y_c = (y - 1.25) / 0.75
    z_c = (z - 1.25) / 0.75
    
    # Construct a diverse feature basis
    # We have 10 parameters. We want to cover:
    # 1. Constant
    # 2-4. Linear (x, y, z)
    # 5-6. Key interactions (xy, xz)
    # 7-10. Non-linear univariate features to capture exp/log/trig shapes
    
    # 1. Constant
    f0 = np.ones_like(x)
    
    # 2-4. Linear terms (centered)
    f1 = x_c
    f2 = y_c
    f3 = z_c
    
    # 5-6. Interaction terms
    f4 = x_c * y_c
    f5 = x_c * z_c
    
    # 7-10. Non-linear univariate terms
    # Exponential growth/decay (centered to keep values moderate)
    f6 = np.exp(x_c)
    
    # Logarithmic behavior (log(x) is smooth on [0.5, 2])
    # Using log(x) directly, scaled/shifted to be roughly centered
    f7 = np.log(x) - np.log(1.25)
    
    # Periodic behavior
    f8 = np.sin(np.pi * x_c)
    
    # Hyperbolic tangent (saturation)
    f9 = np.tanh(x_c)
    
    # Combine
    pred = (p[0] * f0 +
            p[1] * f1 +
            p[2] * f2 +
            p[3] * f3 +
            p[4] * f4 +
            p[5] * f5 +
            p[6] * f6 +
            p[7] * f7 +
            p[8] * f8 +
            p[9] * f9)
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
# EVOLVE-BLOCK-END
