import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation of motion for a driven nonlinear oscillator.
    
    Model:
    a = c0 + c1*x + c2*x^3 + c3*v + c4*v*|v| + c5*x*v 
      + A_add * sin(w*t + phi) + A_par * sin(w*t + phi) * x
    
    Parameters:
    params[0]: c0 (Offset)
    params[1]: c1 (Linear Spring)
    params[2]: c2 (Cubic Spring)
    params[3]: c3 (Linear Damping)
    params[4]: c4 (Quadratic Damping)
    params[5]: c5 (Coupling x*v)
    params[6]: A_add (Additive Drive Amplitude)
    params[7]: A_par (Parametric Drive Amplitude)
    params[8]: w (Frequency)
    params[9]: phi (Phase)
    """
    p = params
    
    # Pre-compute trigonometric term
    phase = p[8] * t + p[9]
    sin_phase = np.sin(phase)
    
    # Terms
    term_offset = p[0]
    term_x = p[1] * x
    term_x3 = p[2] * x**3
    term_v = p[3] * v
    term_v2 = p[4] * v * np.abs(v)
    term_xv = p[5] * x * v
    
    term_drive_add = p[6] * sin_phase
    term_drive_par = p[7] * sin_phase * x
    
    return (term_offset + term_x + term_x3 + term_v + term_v2 + 
            term_xv + term_drive_add + term_drive_par)
