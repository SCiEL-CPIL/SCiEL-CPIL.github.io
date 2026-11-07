import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial model for stress-strain with temperature.
    params: [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]
    """
    s = strain
    t = temp
    p = params
    
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    t2 = t * t
    
    pred = (p[0] + 
            p[1] * s + 
            p[2] * s2 + 
            p[3] * s3 + 
            p[4] * s4 + 
            p[5] * t + 
            p[6] * s * t + 
            p[7] * s2 * t + 
            p[8] * s3 * t + 
            p[9] * t2)
            
    return pred
