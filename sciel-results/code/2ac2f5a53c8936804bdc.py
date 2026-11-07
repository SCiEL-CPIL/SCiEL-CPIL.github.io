import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params.flatten()
    
    # Ensure length is 10
    if len(p) < 10:
        # Fallback to a simple linear model if params are malformed
        # This should rarely happen as the evaluator provides 10 params
        return p[0]*x + p[1]*y + p[2]*z + p[3]
        
    # Construct a complete polynomial of degree 2 in x, y, z
    # Terms: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    # This uses exactly 10 parameters.
    
    term_0 = 1.0
    term_1 = x
    term_2 = y
    term_3 = z
    term_4 = x * x
    term_5 = y * y
    term_6 = z * z
    term_7 = x * y
    term_8 = x * z
    term_9 = y * z
    
    return (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)
