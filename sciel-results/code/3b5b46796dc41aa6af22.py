import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1] range for better conditioning
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
        
    # Basis:
    # 1. Constant
    # 2. Linear u
    # 3. Linear v
    # 4. Linear w
    # 5. Sine u
    # 6. Sine v
    # 7. Sine w
    # 8. Interaction uv
    # 9. Interaction uw
    # 10. Interaction vw
    
    # This basis allows for linear trends, oscillatory behavior, and interactions.
    # It is more flexible than a pure quadratic polynomial and may avoid local minima 
    # associated with polynomial fitting for non-polynomial targets.
    
    return (
        p[0] * 1.0 +
        p[1] * u +
        p[2] * v +
        p[3] * w +
        p[4] * np.sin(u) +
        p[5] * np.sin(v) +
        p[6] * np.sin(w) +
        p[7] * u * v +
        p[8] * u * w +
        p[9] * v * w
    )
