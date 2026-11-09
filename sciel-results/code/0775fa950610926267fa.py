"""Symbolic regression with 1 input - Small Tanh Neural Network with Linear Component."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 2 shifted Tanh neurons, 1 symmetric Tanh neuron, 1 Linear neuron, 1 Bias.
    
    y = c + w1 * tanh(a1 * x + b1) + w2 * tanh(a2 * x + b2) + w3 * tanh(a3 * x) + w4 * x
    
    10 parameters:
      params[0]: c (bias)
      params[1]: w1 (weight for neuron 1)
      params[2]: a1 (slope for neuron 1)
      params[3]: b1 (bias for neuron 1)
      params[4]: w2 (weight for neuron 2)
      params[5]: a2 (slope for neuron 2)
      params[6]: b2 (bias for neuron 2)
      params[7]: w3 (weight for neuron 3, symmetric)
      params[8]: a3 (slope for neuron 3)
      params[9]: w4 (weight for linear term)
    
    This structure explicitly captures linear trends and odd-symmetric non-linearities,
    potentially aiding BFGS convergence from the all-ones initialization compared
    to three fully shifted tanh neurons.
    """
    c = params[0]
    
    # Neuron 1: Shifted Tanh
    w1, a1, b1 = params[1], params[2], params[3]
    h1 = np.tanh(a1 * x + b1)
    
    # Neuron 2: Shifted Tanh
    w2, a2, b2 = params[4], params[5], params[6]
    h2 = np.tanh(a2 * x + b2)
    
    # Neuron 3: Symmetric Tanh (no bias)
    w3, a3 = params[7], params[8]
    h3 = np.tanh(a3 * x)
    
    # Linear Term
    w4 = params[9]
    h4 = x
    
    # Combine
    y = c + w1 * h1 + w2 * h2 + w3 * h3 + w4 * h4
    
    return y
# EVOLVE-BLOCK-END
