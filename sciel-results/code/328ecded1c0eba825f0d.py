"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single Hidden Layer Neural Network (MLP) with Tanh Activation.
    
    Architecture:
    - Input: 2 features (x, y)
    - Hidden Layer: 2 neurons with tanh activation
    - Output Layer: 1 neuron (linear)
    
    Parameters (10 total):
    p0: Weight from x to hidden neuron 1
    p1: Weight from y to hidden neuron 1
    p2: Bias for hidden neuron 1
    p3: Weight from x to hidden neuron 2
    p4: Weight from y to hidden neuron 2
    p5: Bias for hidden neuron 2
    p6: Weight from hidden neuron 1 to output
    p7: Weight from hidden neuron 2 to output
    p8: Bias for output layer
    p9: Unused (set to 0 to avoid affecting gradient, or could be used for scaling if needed, 
        but here we stick to standard 9-param net and ignore p9 for stability)
        
    Actually, let's use all 10 to be safe? 
    A 2-2-1 net has:
    Hidden 1: w_x1, w_y1, b1 (3)
    Hidden 2: w_x2, w_y2, b2 (3)
    Output:   w_h1, w_h2, b_out (3)
    Total: 9.
    
    We have 10 params. We can add a global scaling factor or just ignore the 10th.
    Ignoring the 10th is safer for BFGS if it starts at 1 and doesn't contribute.
    However, if the target requires a specific scale, the output bias/weights handle it.
    
    Let's map:
    p0: w_x1
    p1: w_y1
    p2: b1
    p3: w_x2
    p4: w_y2
    p5: b2
    p6: w_h1
    p7: w_h2
    p8: b_out
    p9: (Unused, effectively 0 contribution)
    
    Model:
    h1 = tanh(p0*x + p1*y + p2)
    h2 = tanh(p3*x + p4*y + p5)
    out = p6*h1 + p7*h2 + p8
    
    This family can approximate:
    - Polynomials (via Taylor expansion of tanh)
    - Multiplicative interactions (x*y approximated by tanh(ax)*tanh(by) if a,b large)
    - Smooth non-polynomial functions
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, _ = params
    
    # Compute hidden layer activations
    # Using np.tanh for element-wise operation
    z1 = p0 * x + p1 * y + p2
    z2 = p3 * x + p4 * y + p5
    
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    
    # Compute output
    result = p6 * h1 + p7 * h2 + p8
    
    return result
# EVOLVE-BLOCK-END
