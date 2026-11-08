"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a small MLP.
    
    Structure:
    - Inputs: x (position), v (velocity)
    - Hidden Layer: 4 units with tanh activation
    - Output Layer: Linear combination of hidden units + bias
    
    Parameters (10 total):
    - p[0:4]: Weights for x to hidden units
    - p[4:8]: Weights for v to hidden units
    - p[8]:   Bias for hidden units (shared)
    - p[9]:   Output bias
    
    The output is a weighted sum of the hidden activations. 
    Note: To strictly use only 10 parameters and maintain a clear mapping, 
    we will use a structure where the hidden layer outputs are combined 
    with fixed, symmetric weights or we allocate params differently.
    
    Let's refine the parameter allocation for a 2-input, 4-hidden, 1-output net with 10 params:
    Standard MLP: W_in (2x4=8), b_in (4=4), W_out (4x1=4), b_out (1=1) -> 17 params. Too many.
    
    Alternative: 
    We can use a "radial basis" or "local basis" approach, or a specific sparse connectivity.
    However, the prompt says "up to 10 fitted constants".
    
    Let's try a different structural family: **Fourier Series in phase space**.
    Nonlinear oscillators often have periodic solutions. A Fourier basis in x and v
    might capture the limit cycle structure better than polynomials.
    
    Basis:
    sin(x), cos(x), sin(v), cos(v), sin(x+v), cos(x-v), x*v, x^2, v^2, 1
    This is a fixed linear basis again. The previous inspirations showed rational functions and tanh-saturated odd powers.
    
    The cell finding mentioned: "Using tanh-saturated odd-power basis functions with scaled inputs reduces feature magnitude disparity... giving BFGS a better-conditioned landscape".
    
    Let's implement a **Tanh-Saturated Polynomial Basis** with explicit scaling.
    This is a structural change from the linear seed. It uses the same "linear combination" interface but with a much richer, bounded, and well-scaled basis.
    
    Basis functions:
    1. x * tanh(x)
    2. x^2 * tanh(x)  -- wait, tanh(x) is odd. x^2*tanh(x) is odd.
    Let's use:
    1. tanh(x)
    2. tanh(x) * x
    3. tanh(v)
    4. tanh(v) * v
    5. tanh(x) * tanh(v)
    6. x * v
    7. x^2
    8. v^2
    9. x * tanh(v)
    10. 1
    
    This provides a mix of saturated and polynomial terms, capturing nonlinearity while keeping values bounded (mostly) and scaled.
    """
    p = params
    
    # Scale inputs to roughly [-1, 1] range to help optimization?
    # Training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
    # They are already small. Scaling might not be necessary, but let's ensure stability.
    
    # Compute basis functions
    # Use np.tanh for saturation
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # 1. Linear-ish saturated
    b0 = p[0] * tanh_x
    
    # 2. Quadratic-ish saturated (odd symmetry)
    b1 = p[1] * (x * tanh_x)
    
    # 3. Velocity saturated
    b2 = p[2] * tanh_v
    
    # 4. Velocity quadratic-ish saturated
    b3 = p[3] * (v * tanh_v)
    
    # 5. Interaction saturated
    b4 = p[4] * (tanh_x * tanh_v)
    
    # 6. Pure interaction
    b5 = p[5] * (x * v)
    
    # 7. Position quadratic
    b6 = p[6] * (x * x)
    
    # 8. Velocity quadratic
    b7 = p[7] * (v * v)
    
    # 9. Cross saturated
    b8 = p[8] * (x * tanh_v)
    
    # 10. Bias
    b9 = p[9]
    
    return b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9
# EVOLVE-BLOCK-END
