import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute Legendre polynomials up to degree 4
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t**2 - 1)
    # P_3(t) = 0.5 * (5*t**3 - 3*t)
    # P_4(t) = 0.125 * (35*t**4 - 30*t**2 + 3)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3.0 * x**2 - 1.0)
    px3 = 0.5 * (5.0 * x**3 - 3.0 * x)
    px4 = 0.125 * (35.0 * x**4 - 30.0 * x**2 + 3.0)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3.0 * y**2 - 1.0)
    py3 = 0.5 * (5.0 * y**3 - 3.0 * y)
    py4 = 0.125 * (35.0 * y**4 - 30.0 * y**2 + 3.0)
    
    # Construct the 10 basis functions
    # We prioritize pure higher-order terms over mixed lower-order terms
    # to capture potential 4th order behavior in x or y individually.
    b0 = px0 * py0  # 1
    b1 = px1 * py0  # x
    b2 = px0 * py1  # y
    b3 = px2 * py0  # P_2(x)
    b4 = px0 * py2  # P_2(y)
    b5 = px3 * py0  # P_3(x)
    b6 = px0 * py3  # P_3(y)
    b7 = px4 * py0  # P_4(x)
    b8 = px0 * py4  # P_4(y)
    b9 = px1 * py1  # x*y
    
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
