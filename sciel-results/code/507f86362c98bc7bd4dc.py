import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    # 3 hidden units
    # w: 3 params (input to hidden)
    # b: 3 params (hidden biases)
    # v: 3 params (hidden to output)
    # c: 1 param (output bias)
    
    w = params[0:3]
    b = params[3:6]
    v = params[6:9]
    c = params[9]
    
    # Add fixed asymmetry to break symmetry of identical hidden units
    # This helps BFGS escape the symmetric basin where all hidden units act as one
    w_asym = w + np.array([0.1, 0.2, 0.3])
    b_asym = b + np.array([0.1, 0.2, 0.3])
    
    # Calculate hidden layer
    z = x[:, np.newaxis] * w_asym[np.newaxis, :] + b_asym[np.newaxis, :]
    h = np.tanh(z)
    
    # Output layer
    pred = h @ v + c
    
    # Safety guard
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
