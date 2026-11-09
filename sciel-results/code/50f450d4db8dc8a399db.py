import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1] for better conditioning
    # x, y, z are in [0.5, 2.0]
    # center = 1.25, scale = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Construct quadratic polynomial in normalized space
    # Terms: 1, u, v, w, u^2, v^2, w^2, uv, uw, vw
    return (p0 + 
            p1 * u + 
            p2 * v + 
            p3 * w + 
            p4 * u**2 + 
            p5 * v**2 + 
            p6 * w**2 + 
            p7 * u * v + 
            p8 * u * w + 
            p9 * v * w)
