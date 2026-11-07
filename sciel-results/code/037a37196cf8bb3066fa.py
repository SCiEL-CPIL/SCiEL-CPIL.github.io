import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric basis functions
    # This basis is bounded and well-conditioned for inputs in [-1, 1]
    # We use frequencies that cover the range [-1, 1] effectively
    
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    
    sin_2px = np.sin(2 * np.pi * x)
    cos_2py = np.cos(2 * np.pi * y)
    
    # Interactions
    sin_px_sin_py = sin_px * sin_py
    cos_px_cos_py = cos_px * cos_py
    sin_px_cos_py = sin_px * cos_py
    
    # Basis vector
    # 0: 1
    # 1: sin(pi x)
    # 2: cos(pi x)
    # 3: sin(pi y)
    # 4: cos(pi y)
    # 5: sin(pi x)sin(pi y)
    # 6: cos(pi x)cos(pi y)
    # 7: sin(2 pi x)
    # 8: cos(2 pi y)
    # 9: sin(pi x)cos(pi y)
    
    f = (params[0] * 1 +
         params[1] * sin_px +
         params[2] * cos_px +
         params[3] * sin_py +
         params[4] * cos_py +
         params[5] * sin_px_sin_py +
         params[6] * cos_px_cos_py +
         params[7] * sin_2px +
         params[8] * cos_2py +
         params[9] * sin_px_cos_py)
         
    return f
