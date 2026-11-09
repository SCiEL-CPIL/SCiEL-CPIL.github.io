"""
Strategy: Two-Layer ReLU Neural Network with Learnable Output Weights.
Instead of a single-layer tanh MLP, we use a deeper network with ReLU activations.
The first layer has 5 units, and the second layer has 3 units. The output is a
weighted sum of the second-layer activations with learnable weights. This allows
for more complex non-linear combinations and better approximation of functions
with multiple local features or sharp transitions.

Params layout (length 10):
params[0:5] -> Weights for the 5 units in the first layer (input to hidden1)
params[5:7] -> Biases for the 3 units in the second layer (hidden1 to hidden2)
params[7:10] -> Weights for the 3 units in the second layer (hidden2 to output)
Note: The biases for the first layer are fixed at 0 to conserve parameters,
and the weights from hidden1 to hidden2 are fixed at 1.0 to keep the parameter
count at exactly 10. This is a constrained architecture, but it allows for
deeper non-linear interactions.

Actually, to better utilize the 10 parameters, let's reconfigure:
params[0:3] -> Weights from input to hidden1 (3 units)
params[3:6] -> Biases for hidden1 (3 units)
params[6:8] -> Weights from hidden1 to hidden2 (2 units)
params[8:10] -> Biases for hidden2 (2 units)
The output is the sum of the hidden2 activations. This gives us a 2-layer
network with 3 and 2 units respectively, and all weights and biases are learnable.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a two-layer ReLU neural network.
    
    params[0:3] are the weights from input to hidden layer 1 (3 units).
    params[3:6] are the biases for hidden layer 1 (3 units).
    params[6:8] are the weights from hidden layer 1 to hidden layer 2 (2 units).
    params[8:10] are the biases for hidden layer 2 (2 units).
    
    The output is the sum of the hidden layer 2 activations.
    """
    # Extract parameters
    w1 = params[0:3]  # (3,)
    b1 = params[3:6]  # (3,)
    w2 = params[6:8]  # (2,)
    b2 = params[8:10] # (2,)
    
    # Ensure x is a column vector for easy broadcasting
    x_reshaped = x.reshape(-1, 1)  # (N, 1)
    
    # First layer: z1 = x * w1 + b1 -> (N, 3)
    w1_reshaped = w1.reshape(1, 3)  # (1, 3)
    b1_reshaped = b1.reshape(1, 3)  # (1, 3)
    z1 = x_reshaped * w1_reshaped + b1_reshaped  # (N, 3)
    a1 = np.maximum(0, z1)  # ReLU activation -> (N, 3)
    
    # Second layer: z2 = a1 * w2 + b2 -> (N, 2)
    # w2 is (2,), but we need to broadcast it correctly.
    # a1 is (N, 3), w2 is (2,). We need to treat w2 as a (3, 2) matrix?
    # No, the parameters are only 2 weights. This implies a fixed connection?
    # Let's re-interpret: w2[0] connects a1[0] to a2[0], w2[1] connects a1[1] to a2[1]?
    # Or is it a dense layer? With only 2 params, it's not dense.
    # Let's assume a simpler structure: The 2nd layer has 2 units.
    # Unit 0 of layer 2: weight = w2[0], bias = b2[0], input = a1[0] + a1[1] + a1[2]? No.
    # Let's make it a fully connected layer but with tied weights or a specific structure.
    # To make it meaningful with 2 weights for 3 inputs, let's use:
    # a2[0] = max(0, w2[0] * (a1[0] + a1[1] + a1[2]) + b2[0])
    # a2[1] = max(0, w2[1] * (a1[0] + a1[1] + a1[2]) + b2[1])
    # This is a bit arbitrary. Let's stick to a standard MLP structure but with
    # fewer parameters by using a smaller network.
    
    # Revised Plan: 2 layers, 2 units in first, 2 units in second.
    # params[0:2] -> Weights input to hidden1 (2 units)
    # params[2:4] -> Biases hidden1 (2 units)
    # params[4:8] -> Weights hidden1 to hidden2 (2x2 matrix)
    # params[8:10] -> Biases hidden2 (2 units)
    # This uses 10 parameters exactly.
    
    w1 = params[0:2]  # (2,)
    b1 = params[2:4]  # (2,)
    w2_matrix = params[4:8].reshape(2, 2)  # (2, 2)
    b2 = params[8:10] # (2,)
    
    # First layer: z1 = x * w1 + b1 -> (N, 2)
    w1_reshaped = w1.reshape(1, 2)  # (1, 2)
    b1_reshaped = b1.reshape(1, 2)  # (1, 2)
    z1 = x_reshaped * w1_reshaped + b1_reshaped  # (N, 2)
    a1 = np.maximum(0, z1)  # ReLU activation -> (N, 2)
    
    # Second layer: z2 = a1 @ w2_matrix + b2 -> (N, 2)
    # a1 is (N, 2), w2_matrix is (2, 2)
    z2 = a1 @ w2_matrix + b2_reshaped  # (N, 2)
    a2 = np.maximum(0, z2)  # ReLU activation -> (N, 2)
    
    # Output: sum of hidden layer 2 activations
    result = np.sum(a2, axis=1)
    
    return result
# EVOLVE-BLOCK-END
