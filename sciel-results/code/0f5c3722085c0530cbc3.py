import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    DEADLINE = t0 + 140.0  # 10s safety margin before 150s kill
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    # Using a Fourier basis for global periodicity and smooth interactions
    # Basis:
    # 0: 1
    # 1: sin(pi*x)
    # 2: sin(pi*y)
    # 3: cos(pi*x)
    # 4: cos(pi*y)
    # 5: sin(pi*x)*sin(pi*y)
    # 6: cos(pi*x)*cos(pi*y)
    # 7: sin(2*pi*x)
    # 8: sin(2*pi*y)
    # 9: cos(2*pi*x)*cos(2*pi*y)
    
    sx1 = np.sin(np.pi * x)
    sy1 = np.sin(np.pi * y)
    cx1 = np.cos(np.pi * x)
    cy1 = np.cos(np.pi * y)
    sx2 = np.sin(2 * np.pi * x)
    sy2 = np.sin(2 * np.pi * y)
    cx2 = np.cos(2 * np.pi * x)
    cy2 = np.cos(2 * np.pi * y)
    
    pred = (p0 + 
            p1 * sx1 + 
            p2 * sy1 + 
            p3 * cx1 + 
            p4 * cy1 + 
            p5 * sx1 * sy1 + 
            p6 * cx1 * cy1 + 
            p7 * sx2 + 
            p8 * sy2 + 
            p9 * cx2 * cy2)
    
    return pred
