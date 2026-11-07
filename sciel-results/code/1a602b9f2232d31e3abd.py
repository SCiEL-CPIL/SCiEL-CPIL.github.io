import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
        p = params
        # Avoid division by zero (inputs are >= 0.5, so safe)
        inv_x = 1.0 / x
        inv_y = 1.0 / y
        inv_z = 1.0 / z
        
        term_0 = 1.0
        term_1 = x
        term_2 = y
        term_3 = z
        term_4 = inv_x
        term_5 = inv_y
        term_6 = inv_z
        term_7 = x * y
        term_8 = y * z
        term_9 = z * x
        
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
