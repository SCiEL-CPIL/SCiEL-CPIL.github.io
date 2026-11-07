import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions: a constant, a linear term, 
    and 4 pairs of sine and cosine functions with increasing frequencies.
    
    This Fourier-like basis is designed to capture periodic or oscillatory 
    behavior in the target function, which polynomial bases (like Legendre or 
    Chebyshev) may approximate poorly if the function has high-frequency 
    components or is inherently non-polynomial (e.g., trigonometric).
    
    Basis:
      0: 1
      1: x
      2: cos(pi/2 * x)
      3: sin(pi/2 * x)
      4: cos(pi * x)
      5: sin(pi * x)
      6: cos(3*pi/2 * x)
      7: sin(3*pi/2 * x)
      8: cos(2*pi * x)
      9: sin(2*pi * x)
    """
    # Define frequencies
    # Using multiples of pi/2 to cover a range of periods
    # Period 4, 2, 4/3, 1
    
    # Compute trigonometric terms
    c1 = np.cos(np.pi / 2.0 * x)
    s1 = np.sin(np.pi / 2.0 * x)
    
    c2 = np.cos(np.pi * x)
    s2 = np.sin(np.pi * x)
    
    c3 = np.cos(3.0 * np.pi / 2.0 * x)
    s3 = np.sin(3.0 * np.pi / 2.0 * x)
    
    c4 = np.cos(2.0 * np.pi * x)
    s4 = np.sin(2.0 * np.pi * x)
    
    # Stack the basis functions
    # Order: 1, x, c1, s1, c2, s2, c3, s3, c4, s4
    B = np.column_stack([
        np.ones_like(x),
        x,
        c1, s1,
        c2, s2,
        c3, s3,
        c4, s4
    ])
    
    # Linear combination
    return B @ params
