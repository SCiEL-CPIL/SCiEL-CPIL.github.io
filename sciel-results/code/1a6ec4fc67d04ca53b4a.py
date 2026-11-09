import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    u = (x - 3.05) / 2.95
    
    # Ensure x is a numpy array and handle potential shape issues
    x = np.asarray(x, dtype=float)
    u = (x - 3.05) / 2.95
    
    # Define 10 basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2. Linear
    b1 = u
    
    # 3. Quadratic
    b2 = u ** 2
    
    # 4. Cubic
    b3 = u ** 3
    
    # 5. Sine
    b4 = np.sin(u)
    
    # 6. Cosine
    b5 = np.cos(u)
    
    # 7. Sine (higher frequency)
    b6 = np.sin(2 * u)
    
    # 8. Cosine (higher frequency)
    b7 = np.cos(2 * u)
    
    # 9. Exponential growth
    b8 = np.exp(u)
    
    # 10. Exponential decay
    b9 = np.exp(-u)
    
    # Combine basis functions with parameters
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
