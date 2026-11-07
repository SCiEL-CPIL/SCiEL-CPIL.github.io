import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using Gaussian RBF basis in (x, v) space
    plus harmonic time terms.
    
    a = A1 * exp(-((x_s - cx1)^2 + (v_s - cv1)^2) / (2*sigma^2)) + 
        A2 * exp(-((x_s - cx2)^2 + (v_s - cv2)^2) / (2*sigma^2)) + 
        p7*(t-40) + p8*cos(t) + p9*sin(t)
        
    Inputs are scaled to O(1) for spatial terms.
    """
    # Scale inputs to O(1) for better conditioning
    # x range [-0.2056, 0.1749] -> scale by 0.2
    # v range [-0.3391, 0.3624] -> scale by 0.35
    
    X = x / 0.2
    V = v / 0.35
    
    # Time centering
    T = t - 40.0
    
    # Parameters:
    # p0: amplitude of RBF1
    # p1: amplitude of RBF2
    # p2: center x of RBF1
    # p3: center v of RBF1
    # p4: center x of RBF2
    # p5: center v of RBF2
    # p6: sigma (width) of RBFs
    # p7: linear time term
    # p8: cos(t) amplitude
    # p9: sin(t) amplitude
    
    A1 = params[0]
    A2 = params[1]
    cx1 = params[2]
    cv1 = params[3]
    cx2 = params[4]
    cv2 = params[5]
    sigma = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 0.01)
    
    # RBF terms
    rbf1 = np.exp(-((X - cx1)**2 + (V - cv1)**2) / (2.0 * sigma_safe**2))
    rbf2 = np.exp(-((X - cx2)**2 + (V - cv2)**2) / (2.0 * sigma_safe**2))
    
    # Time terms
    time_linear = p7 * T
    time_cos = p8 * np.cos(t)
    time_sin = p9 * np.sin(t)
    
    # Combine
    pred = A1 * rbf1 + A2 * rbf2 + time_linear + time_cos + time_sin
    
    return pred

# EVOLVE-BLOCK-END
