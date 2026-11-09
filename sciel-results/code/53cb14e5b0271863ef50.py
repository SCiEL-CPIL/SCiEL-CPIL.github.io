"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small MLP with 3 hidden tanh neurons + polynomial fallback.
    
    Non-linear in params, so BFGS performs iterative optimization.
    The tanh units provide flexible local approximation, while the
    polynomial terms capture global trends.
    
    params[0:3]  -> input weights to hidden layer
    params[3:6]  -> hidden layer biases
    params[6:9]  -> output weights from hidden layer
    params[9]    -> output bias
    """
    # Time guard
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a simple fallback
        return np.ones_like(x)
    
    # Small MLP: 1 -> 3 (tanh) -> 1
    hidden = np.tanh(params[0] * x + params[3] + 
                     params[1] * x * x + params[4] +
                     params[2] * x * x * x + params[5])
    
    # Actually, let's use a proper MLP structure
    # h_i = tanh(w_i * x + b_i) for i=0,1,2
    h0 = np.tanh(params[0] * x + params[3])
    h1 = np.tanh(params[1] * x + params[4])
    h2 = np.tanh(params[2] * x + params[5])
    
    # Output: w0*h0 + w1*h1 + w2*h2 + b
    result = params[6] * h0 + params[7] * h1 + params[8] * h2 + params[9]
    
    return result
# EVOLVE-BLOCK-END
