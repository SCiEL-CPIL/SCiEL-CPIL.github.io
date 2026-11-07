"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Small neural network with 3 hidden units using tanh activation
    # params:
    # p0, p1: weight, bias for unit 0
    # p2, p3: weight, bias for unit 1
    # p4, p5: weight, bias for unit 2
    # p6, p7, p8: output weights for units 0, 1, 2
    # p9: output bias
    
    # Calculate hidden units
    z0 = np.tanh(params[0] * x + params[1])
    z1 = np.tanh(params[2] * x + params[3])
    z2 = np.tanh(params[4] * x + params[5])
    
    # Combine
    return params[6] * z0 + params[7] * z1 + params[8] * z2 + params[9]
# EVOLVE-BLOCK-END
