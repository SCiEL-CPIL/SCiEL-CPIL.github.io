import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 3-hidden-layer tanh network.
    
    The function is modeled as a single-hidden-layer (3 units) tanh network:
    y(x) = c + sum_i v_i * tanh(w_i * x + b_i)
    
    This approach provides universal approximation capability with smooth,
    bounded activations that are well-suited for BFGS optimization. The
    10 parameters are: [w1, w2, w3, b1, b2, b3, v1, v2, v3, c]
    
    Parameters:
    x: Input array
    params: Array of 10 parameters
    """
    # Extract params
    w1, w2, w3 = params[0], params[1], params[2]
    b1, b2, b3 = params[3], params[4], params[5]
    v1, v2, v3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute hidden units
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    h3 = np.tanh(w3 * x + b3)
    
    # Combine
    y = v1 * h1 + v2 * h2 + v3 * h3 + c
    return y
