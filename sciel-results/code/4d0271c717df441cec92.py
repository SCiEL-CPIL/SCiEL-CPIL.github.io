import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute Legendre polynomials up to degree 3
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t**2 - 1)
    # P_3(t) = 0.5 * (5*t**3 - 3*t)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3.0 * x**2 - 1.0)
    px3 = 0.5 * (5.0 * x**3 - 3.0 * x)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3.0 * y**2 - 1.0)
    py3 = 0.5 * (5.0 * y**3 - 3.0 * y)
    
    # Construct the 10 basis functions
    # We prioritize mixed terms of lower order over pure terms of highest order
    # to better capture interactions.
    b0 = px0 * py0  # 1
    b1 = px1 * py0  # x
    b2 = px0 * py1  # y
    b3 = px2 * py0  # P_2(x)
    b4 = px0 * py2  # P_2(y)
    b5 = px1 * py1  # x*y
    b6 = px2 * py1  # P_2(x)*y
    b7 = px1 * py2  # x*P_2(y)
    b8 = px3 * py0  # P_3(x)
    b9 = px0 * py3  # P_3(y)
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
            
    return pred
