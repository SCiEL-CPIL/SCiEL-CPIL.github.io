import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack 10 parameters
    a0, a1, a2, a3, w1, b1, c1, w2, b2, c2 = params
    
    # Polynomial part (Cubic)
    poly = a0 + x * (a1 + x * (a2 + x * a3))
    
    # Stable Sigmoid
    def stable_sigmoid(z):
        # Use np.clip to prevent overflow in exp, or the where method
        # np.where is generally safe if implemented correctly
        z = np.clip(z, -500, 500) # Safety clip
        return 1.0 / (1.0 + np.exp(-z))
        
    sig1 = stable_sigmoid(w1 * x + b1)
    sig2 = stable_sigmoid(w2 * x + b2)
    
    result = poly + c1 * sig1 + c2 * sig2
    
    return result
