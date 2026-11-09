import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a small Multi-Layer Perceptron (MLP) 
    with 2 hidden layers.
    
    Strategy:
    - Uses a 2-4-4-1 neural network architecture.
    - The 10 parameters are distributed as follows:
      - 3 parameters for the first hidden layer: 2 weights (W1_1, W1_2) and 1 bias (b1)
        Wait, a 2-4-4-1 net has:
        Layer 1: 2 inputs, 4 outputs -> 8 weights + 4 biases = 12 params. Too many.
        
    Let's adjust the architecture to fit exactly 10 parameters.
    
    Architecture: 2 inputs -> 2 hidden neurons -> 1 output.
    Layer 1 (Hidden):
      - 2 inputs, 2 outputs.
      - Weights: W1 (2x2) = 4 params.
      - Biases: b1 (2) = 2 params.
      Total so far: 6 params.
    
    Layer 2 (Output):
      - 2 inputs, 1 output.
      - Weights: W2 (1x2) = 2 params.
      - Bias: b2 (1) = 1 param.
      Total so far: 9 params.
      
    We have 1 param left. We can use it as a global scaling factor or a temperature for the activation function.
    Let's use it as a scaling factor for the output to allow BFGS to adjust the magnitude if needed, 
    or better, let's make the activation function's steepness learnable.
    
    Revised Plan:
    Hidden Layer 1: 2 neurons.
    Weights: W11, W12, W21, W22 (4 params)
    Biases: b11, b12 (2 params)
    Activation: tanh(steepness * z). Steepness is param 7.
    
    Output Layer: 1 neuron.
    Weights: W1_out, W2_out (2 params)
    Bias: b_out (1 param)
    
    Total: 4 + 2 + 1 + 2 + 1 = 10 params.
    
    This allows the network to learn non-linear interactions between x and y.
    The tanh activation provides smooth, bounded non-linearity which is well-conditioned for BFGS.
    The learnable steepness allows the network to adapt the "sharpness" of the hidden units.
    """
    
    p = params
    
    # Extract parameters
    # Indices 0-3: Weights for hidden layer (2x2 matrix)
    w11 = p[0]
    w12 = p[1]
    w21 = p[2]
    w22 = p[3]
    
    # Indices 4-5: Biases for hidden layer
    b1 = p[4]
    b2 = p[5]
    
    # Index 6: Steepness of the activation function
    steepness = p[6]
    
    # Indices 7-8: Weights for output layer (from 2 hidden to 1 output)
    w1_out = p[7]
    w2_out = p[8]
    
    # Index 9: Bias for output layer
    b_out = p[9]
    
    # Compute Hidden Layer 1
    # h1 = tanh(steepness * (w11*x + w12*y + b1))
    # h2 = tanh(steepness * (w21*x + w22*y + b2))
    
    z1 = w11 * x + w12 * y + b1
    z2 = w21 * x + w22 * y + b2
    
    # Clip steepness * z to prevent overflow in tanh, though tanh is stable.
    # However, if steepness is very large, we might lose gradient info.
    # BFGS starts at 1, so it should be fine.
    
    h1 = np.tanh(steepness * z1)
    h2 = np.tanh(steepness * z2)
    
    # Compute Output Layer
    # out = w1_out * h1 + w2_out * h2 + b_out
    
    return w1_out * h1 + w2_out * h2 + b_out
