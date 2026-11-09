import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0] * 1
    # params[1] * x
    # params[2] * y
    # params[3] * x**2
    # params[4] * y**2
    # params[5] * x*y
    # params[6] * x**3
    # params[7] * y**3
    # params[8] * x**2*y
    # params[9] * x*y**2
    
    return (params[0] + 
            params[1]*x + 
            params[2]*y + 
            params[3]*x**2 + 
            params[4]*y**2 + 
            params[5]*x*y + 
            params[6]*x**3 + 
            params[7]*y**3 + 
            params[8]*x**2*y + 
            params[9]*x*y**2)
