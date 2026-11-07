import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a state-space model with
    time-varying coefficients from a low-order Fourier series in t.
    
    The effective stiffness and damping vary periodically with time,
    allowing the model to capture parametric excitation and non-stationary
    oscillator behavior without high-degree polynomial ill-conditioning.
    
    Model: y = a(t) * x + b(t) * v + c(t)
    where a(t), b(t), c(t) are Fourier series in t with fitted coefficients.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 fitted constants
    
    Returns:
    y: acceleration
    """
    # Scale inputs to O(1) to improve conditioning for BFGS
    x_scale = 0.2
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Fourier basis in time: fundamental frequency and harmonics
    # t is in [30, 50], so we use radian-based harmonics
    # Use a base frequency that captures the natural period of the oscillator
    # For t in [30, 50], a period of ~2*pi gives ~3 cycles
    w = 1.0  # base angular frequency
    
    st = np.sin(t * w)
    ct = np.cos(t * w)
    
    # Unpack 10 parameters:
    # p0, p1: constant and linear time terms for stiffness a(t)
    # p2, p3: constant and linear time terms for damping b(t)  
    # p4, p5: Fourier coefficients for periodic stiffness modulation
    # p6, p7: Fourier coefficients for periodic damping modulation
    # p8, p9: constant and time-varying forcing terms
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Time-varying stiffness: a(t) = p0 + p1*t + p4*sin(t) + p5*cos(t)
    a_t = p0 + p1 * t + p4 * st + p5 * ct
    
    # Time-varying damping: b(t) = p2 + p3*t + p6*sin(t) + p7*cos(t)
    b_t = p2 + p3 * t + p6 * st + p7 * ct
    
    # Forcing term: c(t) = p8 + p9*sin(t)
    c_t = p8 + p9 * st
    
    # Acceleration: y = a(t)*x + b(t)*v + c(t)
    result = a_t * xs + b_t * vs + c_t
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
        
    return result
