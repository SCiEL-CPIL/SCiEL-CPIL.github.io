"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Single Hidden Unit Neural Network with linear skip connections.
    # This family (non-linear basis function expansion) is distinct from global polynomials.
    # It can capture local non-linearities and interactions effectively.
    
    p = params
    
    # 1. Compute the linear combination for the hidden unit
    # p[0], p[1], p[2] are weights for x, y, z
    # p[3] is the bias
    # p[9] is a scaling factor for the input to the activation function to control steepness
    u = p[9] * (p[0] * x + p[1] * y + p[2] * z + p[3])
    
    # 2. Apply non-linear activation (tanh is bounded and smooth)
    h = np.tanh(u)
    
    # 3. Combine hidden unit output with linear skip connections
    # p[4] is the weight for the hidden unit
    # p[5], p[6], p[7] are weights for x, y, z skip connections
    # p[8] is the output bias
    return p[4] * h + p[5] * x + p[6] * y + p[7] * z + p[8]
# EVOLVE-BLOCK-END
