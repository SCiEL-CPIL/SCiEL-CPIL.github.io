import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Fourier series basis to represent the function.
    Basis functions are orthogonal on the uniform domain [-2, 2],
    leading to a well-conditioned design matrix for BFGS.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 coefficients
    
    Returns:
    pred: Array of predictions of shape (N,)
    """
    # 10 parameters:
    # p0: DC term (1)
    # p1: sin(pi/2 * x)
    # p2: cos(pi/2 * x)
    # p3: sin(pi * x)
    # p4: cos(pi * x)
    # p5: sin(3*pi/2 * x)
    # p6: cos(3*pi/2 * x)
    # p7: sin(2*pi * x)
    # p8: cos(2*pi * x)
    # p9: sin(5*pi/2 * x)
    
    # Precompute frequencies
    # k = 0.5, 1.0, 1.5, 2.0, 2.5
    # We use k=0.5, 1, 1.5, 2, 2.5 for sines/cosines where possible.
    
    # To be safe and symmetric, let's use:
    # 1
    # sin(1*x), cos(1*x)  -- freq 1
    # sin(2*x), cos(2*x)  -- freq 2
    # sin(3*x), cos(3*x)  -- freq 3
    # sin(4*x)            -- freq 4
    # This uses integer frequencies, which are orthogonal on [-pi, pi] or similar.
    # On [-2, 2], integer frequencies are not perfectly orthogonal unless the domain is a multiple of pi.
    # 4 is not a multiple of pi.
    # However, the condition number will still be much better than Chebyshev on uniform grid.
    
    # Let's stick to the orthogonal frequencies for domain [-2, 2]:
    # omega_k = k * pi / 2
    # k=1: pi/2
    # k=2: pi
    # k=3: 3*pi/2
    # k=4: 2*pi
    # k=5: 5*pi/2
    
    # Basis:
    # 1
    # sin(pi/2 x), cos(pi/2 x)
    # sin(pi x), cos(pi x)
    # sin(3*pi/2 x), cos(3*pi/2 x)
    # sin(2*pi x), cos(2*pi x)
    # That's 1 + 2 + 2 + 2 + 2 = 9 functions.
    # We have 10 params. Add sin(5*pi/2 x).
    
    t = x
    b0 = np.ones_like(t)
    
    w1 = np.pi / 2.0
    b1 = np.sin(w1 * t)
    b2 = np.cos(w1 * t)
    
    w2 = np.pi
    b3 = np.sin(w2 * t)
    b4 = np.cos(w2 * t)
    
    w3 = 3.0 * np.pi / 2.0
    b5 = np.sin(w3 * t)
    b6 = np.cos(w3 * t)
    
    w4 = 2.0 * np.pi
    b7 = np.sin(w4 * t)
    b8 = np.cos(w4 * t)
    
    w5 = 5.0 * np.pi / 2.0
    b9 = np.sin(w5 * t)
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
