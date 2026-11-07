import numpy as np

def equation(x, params):
    """
    Linear combination of 10 basis functions.
    Basis: [1, x, x^2, x^3, x^4, x^5, sin(x), cos(x), tanh(x), exp(-x^2)]
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure p is 1D and length 10
    if p.shape != (10,):
        p = np.resize(p, 10)
        
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    b5 = x**5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.tanh(x)
    b9 = np.exp(-x**2)
    
    # Combine linearly
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + 
         p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
         
    return y
