"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Fourier Series basis.
    
    This model assumes the dynamics can be approximated by a sum of sinusoidal
    terms of the state variables (x, v) and their interactions. This is often
    more effective for periodic or bounded nonlinear systems than pure polynomials
    because it naturally handles the oscillatory nature of the target.
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Model structure:
        p0 * sin(x)
        p1 * cos(x)
        p2 * sin(v)
        p3 * cos(v)
        p4 * sin(x + v)
        p5 * cos(x + v)
        p6 * sin(x - v)
        p7 * cos(x - v)
        p8 * sin(2*x)
        p9 * cos(2*x)
        
    Note: We use a mix of single and double frequency terms to capture 
    harmonics and phase interactions. The constant term is omitted as it's 
    often absorbed or less critical for acceleration in centered oscillators, 
    but we keep 10 terms for flexibility.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    sin_xv_plus = np.sin(x + v)
    cos_xv_plus = np.cos(x + v)
    sin_xv_minus = np.sin(x - v)
    cos_xv_minus = np.cos(x - v)
    
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    return (
        p0 * sin_x +
        p1 * cos_x +
        p2 * sin_v +
        p3 * cos_v +
        p4 * sin_xv_plus +
        p5 * cos_xv_plus +
        p6 * sin_xv_minus +
        p7 * cos_xv_minus +
        p8 * sin_2x +
        p9 * cos_2x
    )

# EVOLVE-BLOCK-END
