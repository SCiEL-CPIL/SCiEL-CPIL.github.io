import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Basis: 1, x, y, x^2, y^2, xy, sin(x), sin(y), cos(x), cos(y)
    # This matches Inspiration 3 which had the best score.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.cos(x)
    b9 = np.cos(y)
    
    pred = (p[0]*b0 + p[1]*b1 + p[2]*b2 + p[3]*b3 + p[4]*b4 + 
            p[5]*b5 + p[6]*b6 + p[7]*b7 + p[8]*b8 + p[9]*b9)
            
    if not np.all(np.isfinite(pred)):
        # Fallback
        return np.zeros_like(x)
        
    return pred
