import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator with bounded nonlinearities.
    
    Model:
    a = -k(x) * x - c(v) * v + F(t)
    
    Where:
    k(x) = 1 + p0 * tanh(x / scale_x)   (bounded stiffness nonlinearity)
    c(v) = 1 + p1 * tanh(v / scale_v)   (bounded damping nonlinearity)
    F(t) = p2 * sin(p3 * t + p4) + p5   (driven term)
    
    Additional terms:
    + p6 * x * v  (coupling)
    + p7 * x^2 * sign(v) * |v|^0.5 (asymmetric damping)
    + p8 * cos(p3 * t + p4) * x (parametric drive)
    + p9 (offset)
    
    This structure enforces physical stability by using bounded tanh functions
    for stiffness and damping, preventing unbounded polynomial growth.
    """
    # Normalize inputs
    scale_x = 0.21
    scale_v = 0.36
    X = x / scale_x
    V = v / scale_v
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear stiffness: k(x) = 1 + p0 * tanh(x)
    k_x = 1.0 + p0 * np.tanh(X)
    
    # Bounded nonlinear damping: c(v) = 1 + p1 * tanh(v)
    c_v = 1.0 + p1 * np.tanh(V)
    
    # Base restoring and damping forces
    restoring = -k_x * X
    damping = -c_v * V
    
    # Drive terms
    drive_freq = p3
    drive_phase = p4
    drive_sin = np.sin(drive_freq * t + drive_phase)
    drive_cos = np.cos(drive_freq * t + drive_phase)
    
    # Primary drive
    F_t = p2 * drive_sin + p5
    
    # Parametric drive (modulates stiffness with time)
    param_drive = p8 * drive_cos * X
    
    # Coupling term
    coupling = p6 * X * V
    
    # Asymmetric nonlinear damping (soft sign-based)
    # Use |v|^0.5 * sign(v) for smooth-ish asymmetric damping
    abs_v = np.abs(V)
    sign_v = np.sign(V)
    asym_damp = p7 * X * sign_v * np.sqrt(abs_v + 1e-10)
    
    res = restoring + damping + F_t + param_drive + coupling + asym_damp + p9
    
    return res
