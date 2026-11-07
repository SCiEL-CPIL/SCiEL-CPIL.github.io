import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Use a small neural network structure with 2 hidden units (tanh activation)
    and a direct linear path. This provides non-linear capacity to approximate
    complex functions while the linear path helps with convergence.
    
    Parameters (10 total):
    p[0], p[1], p[2]: Weights and bias for hidden unit 1
    p[3], p[4], p[5]: Weights and bias for hidden unit 2
    p[6]: Output weight for hidden unit 1
    p[7]: Output weight for hidden unit 2
    p[8]: Coefficient for direct linear term x
    p[9]: Final bias
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure we have exactly 10 parameters
    if p.shape[0] != 10:
        p_new = np.ones(10, dtype=np.float64)
        n = min(p.shape[0], 10)
        p_new[:n] = p[:n]
        p = p_new
    
    # Calculate hidden layer activations
    # Hidden unit 1
    z1 = np.tanh(p[0] * x + p[1] * y + p[2])
    # Hidden unit 2
    z2 = np.tanh(p[3] * x + p[4] * y + p[5])
    
    # Combine hidden units with a direct linear term and bias
    # The direct term p[8]*x helps capture linear components that might be
    # difficult for the tanh units to isolate from the start.
    result = p[6] * z1 + p[7] * z2 + p[8] * x + p[9]
    
    return result
