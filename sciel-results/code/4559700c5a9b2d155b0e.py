import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier Series basis.
    
    The function is modeled as a sum of sinusoidal terms with increasing frequencies.
    This approach is particularly effective for periodic or smooth oscillatory functions
    and avoids the local minima and scaling issues often encountered with tanh-based neural networks.
    
    y(x) = c0 + c1*sin(x) + c2*cos(x) + c3*sin(2x) + c4*cos(2x) + c5*sin(3x) + c6*cos(3x) + c7*sin(4x) + c8*cos(4x) + c9*sin(5x)
    
    Parameters:
    x: Input array
    params: Array of 10 coefficients [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Compute Fourier terms
    # Using direct computation for speed and stability
    y = c0 + c1 * np.sin(x) + c2 * np.cos(x) + \
        c3 * np.sin(2 * x) + c4 * np.cos(2 * x) + \
        c5 * np.sin(3 * x) + c6 * np.cos(3 * x) + \
        c7 * np.sin(4 * x) + c8 * np.cos(4 * x) + \
        c9 * np.sin(5 * x)
        
    return y
