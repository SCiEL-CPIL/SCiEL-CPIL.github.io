# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single-hidden-layer neural network with tanh activation.
    
    Parameter layout (10 params):
    - p[0], p[1], p[2]: weights from input (x, y, z) to hidden neuron
    - p[3]: bias for hidden neuron
    - p[4], p[5]: weights from hidden neuron to output (two output paths)
    - p[6]: bias for output
    - p[7], p[8], p[9]: additional linear terms (x, y, z) to help gradient flow
    
    Architecture:
    hidden = tanh(p[0]*x + p[1]*y + p[2]*z + p[3])
    output = p[4]*hidden + p[5]*tanh(hidden) + p[6] + p[7]*x + p[8]*y + p[9]*z
    
    This provides a non-linear, universal approximator structure that
    differs structurally from polynomial, trigonometric, and rational bases.
    """
    p = params
    
    # Hidden layer: single tanh neuron
    h_input = p[0] * x + p[1] * y + p[2] * z + p[3]
    hidden = np.tanh(h_input)
    
    # Output: combination of hidden activations and direct linear terms
    # Using two different transforms of hidden to increase expressiveness
    out = (p[4] * hidden + p[5] * np.tanh(2.0 * hidden) + p[6] +
           p[7] * x + p[8] * y + p[9] * z)
    
    return out
# EVOLVE-BLOCK-END
