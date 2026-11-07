import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Shallow Neural Network with 3 hidden units (tanh activation)
    # p[0:3] are input weights (slopes)
    # p[3:6] are hidden biases
    # p[6:9] are output weights
    # p[9] is output bias
    
    v1, v2, v3 = p[0], p[1], p[2]
    u1, u2, u3 = p[3], p[4], p[5]
    w1, w2, w3 = p[6], p[7], p[8]
    b = p[9]
    
    # Compute hidden layer activations
    z1 = np.tanh(v1 * x + u1)
    z2 = np.tanh(v2 * x + u2)
    z3 = np.tanh(v3 * x + u3)
    
    # Output layer
    return w1 * z1 + w2 * z2 + w3 * z3 + b
