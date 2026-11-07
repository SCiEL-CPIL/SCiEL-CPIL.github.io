import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: 10 parameters for a small 3->2->1 tanh network
    # Layout: [w11, w12, w13, w21, w22, w23, w31, w32, b1, b2]
    # Input features: [t_centered, x, v]
    # Hidden layer: 2 units with tanh
    # Output: linear combination of hidden units
    
    # Center and scale inputs for better conditioning
    t_c = (t - 40.0) / 10.0   # t in [30,50] -> [-1, 1]
    x_c = x / 0.5             # x in [-0.34, 0.36] -> ~[-0.68, 0.72]
    v_c = v / 0.5             # v in [-0.21, 0.17] -> ~[-0.42, 0.35]
    
    # Extract weights
    w11, w12, w13 = params[0], params[1], params[2]
    w21, w22, w23 = params[3], params[4], params[5]
    w31, w32 = params[6], params[7]
    b1, b2 = params[8], params[9]
    
    # Hidden layer (2 units)
    h1 = np.tanh(w11 * t_c + w12 * x_c + w13 * v_c + 0.0)
    h2 = np.tanh(w21 * t_c + w22 * x_c + w23 * v_c + 0.0)
    
    # Output
    result = w31 * h1 + w32 * h2
    
    return result
