import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Bounded, smooth basis functions
    t1 = np.tanh(2.0 * x)
    t2 = np.tanh(2.0 * y)
    t3 = np.tanh(2.0 * x * y)
    t4 = np.tanh(x + y)
    t5 = np.tanh(x - y)
    t6 = np.tanh(x * y * (x + y))
    
    pred = (p[0] + p[1] * x + p[2] * y + p[3] * x * y +
            p[4] * t1 + p[5] * t2 + p[6] * t3 +
            p[7] * t4 + p[8] * t5 + p[9] * t6)
    
    return pred
