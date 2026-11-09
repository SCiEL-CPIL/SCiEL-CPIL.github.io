import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Fourier Basis on [-1, 1] with period 2
    # Fundamental frequency is pi
    
    # Pre-compute trigonometric terms
    cos_pi_x = np.cos(np.pi * x)
    sin_pi_x = np.sin(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    sin_pi_y = np.sin(np.pi * y)
    
    cos_2pi_x = np.cos(2 * np.pi * x)
    sin_2pi_x = np.sin(2 * np.pi * x)
    cos_2pi_y = np.cos(2 * np.pi * y)
    sin_2pi_y = np.sin(2 * np.pi * y)
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear-like x (Fundamental)
    b1 = cos_pi_x
    b2 = sin_pi_x
    
    # 3. Linear-like y (Fundamental)
    b3 = cos_pi_y
    b4 = sin_pi_y
    
    # 4. Interactions (Fundamental)
    b5 = cos_pi_x * cos_pi_y
    b6 = sin_pi_x * sin_pi_y
    
    # 5. Higher frequency x
    b7 = cos_2pi_x
    b8 = sin_2pi_x
    
    # 6. Higher frequency y
    b9 = cos_2pi_y
    
    # Note: We skipped sin_2pi_y to save a slot for interactions which are often more important in 2D.
    # Or we could swap b9 with sin_2pi_y? 
    # Let's stick with this mix. It covers x, y, and x*y interactions well.
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
