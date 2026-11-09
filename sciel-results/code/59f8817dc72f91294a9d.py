import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier series basis on [-2, 2] with fundamental frequency pi/2
    # p0: bias
    # p1, p2: freq 1
    # p3, p4: freq 2
    # p5, p6: freq 3
    # p7, p8: freq 4
    # p9: linear term
    
    theta = np.pi / 2.0 * x
    
    c1 = np.cos(theta)
    s1 = np.sin(theta)
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    c3 = np.cos(3 * theta)
    s3 = np.sin(3 * theta)
    c4 = np.cos(4 * theta)
    s4 = np.sin(4 * theta)
    
    return (params[0] + 
            params[1] * c1 + params[2] * s1 + 
            params[3] * c2 + params[4] * s2 + 
            params[5] * c3 + params[6] * s3 + 
            params[7] * c4 + params[8] * s4 + 
            params[9] * x)
