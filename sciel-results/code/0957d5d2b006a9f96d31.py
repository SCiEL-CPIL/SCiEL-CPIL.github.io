"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to t in [0, 1] for numerical stability
    # x_min = 0.1, x_max = 6.0
    # t = (x - 0.1) / (6.0 - 0.1)
    t = (x - 0.1) / 5.9
    
    # Construct basis functions
    # 1. Constant
    phi_0 = np.ones_like(t)
    # 2. Linear
    phi_1 = t
    # 3. Quadratic
    phi_2 = t * t
    # 4. Exponential growth
    phi_3 = np.exp(2.0 * t)
    # 5. Exponential decay
    phi_4 = np.exp(-2.0 * t)
    # 6. Sine freq 1
    phi_5 = np.sin(2.0 * np.pi * t)
    # 7. Cosine freq 1
    phi_6 = np.cos(2.0 * np.pi * t)
    # 8. Sine freq 2
    phi_7 = np.sin(4.0 * np.pi * t)
    # 9. Cosine freq 2
    phi_8 = np.cos(4.0 * np.pi * t)
    # 10. Rational (shifted to avoid singularity, though t=0 is fine for 1/(t+0.1))
    # Using 1/(t + 0.1) to model behavior near x=0.1
    phi_9 = 1.0 / (t + 0.1)
    
    # Combine linearly
    # params is length 10
    y = (params[0] * phi_0 + 
         params[1] * phi_1 + 
         params[2] * phi_2 + 
         params[3] * phi_3 + 
         params[4] * phi_4 + 
         params[5] * phi_5 + 
         params[6] * phi_6 + 
         params[7] * phi_7 + 
         params[8] * phi_8 + 
         params[9] * phi_9)
         
    return y
# EVOLVE-BLOCK-END
