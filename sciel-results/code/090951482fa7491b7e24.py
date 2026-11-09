"""
Non-linear parametric model for symbolic regression.
Uses 10 parameters to define a flexible function combining polynomial,
trigonometric, sigmoidal, and Gaussian components.
"""
import numpy as np

def equation(x, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = params  # length 10
    
    # Decompose parameters into meaningful components
    # p[0:3] -> polynomial coefficients (a0, a1, a2)
    # p[3:5] -> trigonometric (A_sin, w_sin)
    # p[5:7] -> trigonometric (A_cos, w_cos)
    # p[7]   -> sigmoid amplitude
    # p[8]   -> Gaussian amplitude
    # p[9]   -> Gaussian center
    
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    
    A_sin = p[3]
    w_sin = p[4]
    
    A_cos = p[5]
    w_cos = p[6]
    
    A_sig = p[7]
    
    A_gauss = p[8]
    c_gauss = p[9]
    
    # Polynomial part
    y = a0 + a1 * x + a2 * x**2
    
    # Sine component
    y = y + A_sin * np.sin(w_sin * x)
    
    # Cosine component
    y = y + A_cos * np.cos(w_cos * x)
    
    # Sigmoid component (using x/2 as a reasonable scale for [-2,2])
    z = x / 2.0
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    y = y + A_sig * sigmoid
    
    # Gaussian component
    y = y + A_gauss * np.exp(-0.5 * ((x - c_gauss) / 1.0)**2)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
