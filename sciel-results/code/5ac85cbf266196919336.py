import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    1-layer Neural Network with 4 tanh neurons and learnable input scaling.
    
    Parameters (10 total):
      - params[0]: output bias (w_0)
      - params[1]: input scaling factor (s)
      - params[2:6]: weights for the 4 neurons (w_1 to w_4)
      - params[6:10]: biases for the 4 neurons (b_1 to b_4)
    
    Model: y(x) = w_0 + sum_{i=1}^4 w_i * tanh(s * x + b_i)
    """
    w0 = params[0]
    s = params[1]
    w = params[2:6]
    b = params[6:10]
    
    # Compute hidden layer activations
    # x shape: (N,)
    # s, b shape: (1,)
    # z shape: (N, 4)
    z = s * x[:, np.newaxis] + b[np.newaxis, :]
    h = np.tanh(z)
    
    # Linear combination
    result = w0 + h @ w
    
    return result
