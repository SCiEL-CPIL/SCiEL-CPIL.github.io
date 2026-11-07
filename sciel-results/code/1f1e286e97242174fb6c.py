import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: [w1, w2, w3, c1, c2, c3, s1, s2, s3, bias]
    # 3 Gaussian basis functions in (x, v) state space
    # Each basis: weight * exp(-((x-c)^2/s1^2 + (v-c)^2/s2^2))
    # Plus a linear restoring/damping term for stability
    
    w1, w2, w3, c1, c2, c3, s1, s2, s3, bias = params
    
    # Scale inputs to O(1) for numerical stability
    # x range: [-0.34, 0.36] -> center ~0, scale ~0.35
    # v range: [-0.21, 0.17] -> center ~0, scale ~0.2
    x_s = x / 0.35
    v_s = v / 0.20
    
    # Ensure scales are positive and not too small (avoid division by zero / overflow)
    s1 = np.maximum(np.abs(s1), 0.01)
    s2 = np.maximum(np.abs(s2), 0.01)
    s3 = np.maximum(np.abs(s3), 0.01)
    
    # Gaussian basis functions in scaled state space
    # Each basis centered at different points in (x, v) space
    # Basis 1: center at (c1, 0)
    dx1 = x_s - c1
    dv1 = v_s
    g1 = np.exp(-(dx1**2 / s1**2 + dv1**2 / s1**2))
    
    # Basis 2: center at (-c1, 0) - symmetric
    dx2 = x_s + c1
    dv2 = v_s
    g2 = np.exp(-(dx2**2 / s2**2 + dv2**2 / s2**2))
    
    # Basis 3: center at (0, c2) - velocity dependent
    dx3 = x_s
    dv3 = v_s - c2
    g3 = np.exp(-(dx3**2 / s3**2 + dv3**2 / s3**2))
    
    # Combine Gaussian terms
    result = w1 * g1 + w2 * g2 + w3 * g3
    
    # Add linear terms for better conditioning and capturing main dynamics
    # Linear restoring and damping
    result = result - 0.5 * x_s + 0.1 * v_s
    
    # Add bias
    result = result + bias
    
    # Scale back to original units
    # The target acceleration is in range [-0.71, 0.66]
    # Our scaled result should map to this range
    result = result * 0.5
    
    return result
