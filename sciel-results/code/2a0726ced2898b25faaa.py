import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions - rich set to cover many possible target functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    b5 = np.exp(x)
    b6 = np.exp(-x)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.log(x)
    
    # Linear combination
    result = p0*b0 + p1*b1 + p2*b2 + p3*b3 + p4*b4 + p5*b5 + p6*b6 + p7*b7 + p8*b8 + p9*b9
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
