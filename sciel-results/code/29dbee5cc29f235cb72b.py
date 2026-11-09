import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form equation for a damped nonlinear oscillator with periodic driving.
    
    Model:
    y = A * sin(w * t + phi) + k1*x + k2*x^2 + k3*x^3 + c1*v + c2*v^2 + c3*v^3 + kxv*x*v
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        params[0]: Drive Amplitude (A)
        params[1]: Drive Frequency (w)
        params[2]: Drive Phase (phi)
        params[3]: Linear Spring Coeff (k1)
        params[4]: Quadratic Spring Coeff (k2)
        params[5]: Cubic Spring Coeff (k3)
        params[6]: Linear Damping Coeff (c1)
        params[7]: Quadratic Damping Coeff (c2)
        params[8]: Cubic Damping Coeff (c3)
        params[9]: Cross-term Coeff (kxv)
    """
    # Unpack parameters for readability
    A = params[0]
    w = params[1]
    phi = params[2]
    k1 = params[3]
    k2 = params[4]
    k3 = params[5]
    c1 = params[6]
    c2 = params[7]
    c3 = params[8]
    kxv = params[9]
    
    # Compute terms
    drive = A * np.sin(w * t + phi)
    spring = k1 * x + k2 * x**2 + k3 * x**3
    damping = c1 * v + c2 * v**2 + c3 * v**3
    cross = kxv * x * v
    
    return drive + spring + damping + cross
