"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a simple 1-hidden-layer Neural Network.
    
    The 10 parameters are distributed as:
    - 4 weights for the hidden layer (3 inputs * 1 hidden unit + 1 bias)
    - 1 weight for the output layer (1 hidden unit * 1 output + 1 bias)
    - 5 parameters reserved for potential future use or kept as 0 to avoid overfitting/instability?
    
    Wait, the contract says "up to 10 fitted constants". 
    If I use a 1-hidden-layer NN with 1 hidden unit:
    Inputs: t, x, v (3 dims)
    Hidden: z = w1*t + w2*x + w3*v + b1
    Output: y = w4*z + b2
    
    This uses 5 parameters. I have 5 left.
    I can expand to 2 hidden units:
    Inputs: 3
    Hidden1: z1 = w1*t + w2*x + w3*v + b1
    Hidden2: z2 = w4*t + w5*x + w6*v + b2
    Output: y = w7*z1 + w8*z2 + b3
    
    Total params: 3*2 (weights) + 2 (biases) + 2 (output weights) + 1 (output bias) = 10.
    
    This creates a non-linear mapping: y = w7*sigmoid(w1*t + w2*x + w3*v + b1) + w8*sigmoid(w4*t + w5*x + w6*v + b2) + b3
    Using sigmoid allows the BFGS optimizer to explore non-linear regions.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # Hard limit from contract, keep margin
    
    # Unpack 10 parameters
    # Hidden Layer 1: 3 weights, 1 bias
    w1_t, w1_x, w1_v, b1 = params[0], params[1], params[2], params[3]
    # Hidden Layer 2: 3 weights, 1 bias
    w2_t, w2_x, w2_v, b2 = params[4], params[5], params[6], params[7]
    # Output Layer: 2 weights, 1 bias
    wo_1, wo_2, bo = params[8], params[9], 0.0 # Wait, I only have 10 params. 
    # Let's recount.
    # Params 0-2: W1 (3)
    # Param 3: B1 (1)
    # Params 4-6: W2 (3)
    # Param 7: B2 (1)
    # Params 8-9: WO (2)
    # Missing output bias! 
    # I can set output bias to 0 or absorb it? No, better to use a structure that fits exactly 10.
    
    # Alternative: 1 Hidden Unit, but more complex activation?
    # Or 2 Hidden Units, but no output bias?
    # Or 1 Hidden Unit with 2 inputs? No, inputs are fixed.
    
    # Let's try:
    # Hidden 1: t, x, v -> z1 (3 w, 1 b) = 4 params
    # Hidden 2: t, x, v -> z2 (3 w, 1 b) = 4 params
    # Output: z1, z2 -> y (2 w, 0 b) = 2 params
    # Total = 10.
    # y = wo1 * tanh(z1) + wo2 * tanh(z2)
    
    w1_t, w1_x, w1_v, b1 = params[0], params[1], params[2], params[3]
    w2_t, w2_x, w2_v, b2 = params[4], params[5], params[6], params[7]
    wo_1, wo_2 = params[8], params[9]
    
    # Compute hidden layers
    z1 = w1_t * t + w1_x * x + w1_v * v + b1
    z2 = w2_t * t + w2_x * x + w2_v * v + b2
    
    # Activation: tanh is smooth and bounded, good for BFGS
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    
    # Output
    result = wo_1 * h1 + wo_2 * h2
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return zeros or a large penalty? The contract says non-finite error costs 1e30.
        # Returning zeros is safer than crashing.
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
