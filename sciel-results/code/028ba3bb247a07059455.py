"""Symbolic regression: nonlinear oscillator acceleration via physics-informed Fourier-Poly basis.

This program defines a closed-form expression for the acceleration of a nonlinear oscillator.
The basis combines polynomial terms in position (x) and velocity (v) to capture state-dependent
nonlinearities (stiffness and damping) with Fourier series terms in time (t) to capture
periodic forcing and higher harmonic responses.

Basis Terms (10 parameters):
1.  Bias
2.  Linear stiffness (x)
3.  Cubic stiffness (x^3)
4.  Linear damping (v)
5.  Nonlinear damping (x^2 * v)
6.  Primary forcing sine (sin(t))
7.  Primary forcing cosine (cos(t))
8.  Secondary forcing sine (sin(2t))
9.  Secondary forcing cosine (cos(2t))
10. Tertiary forcing sine (sin(3t))
"""
# EVOLVE-BLOCK-START
import numpy as np

# EVOLVE-BLOCK-END

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction of acceleration.
    
    Args:
        t: Time array
        x: Position array
        v: Velocity array
        params: Array of 10 fitted constants
        
    Returns:
        Array of predicted accelerations
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = len(t)
    
    # State-dependent terms (Nonlinear oscillator dynamics)
    f0 = np.ones(n)          # Bias
    f1 = x                   # Linear spring
    f2 = x ** 3              # Cubic spring (Duffing)
    f3 = v                   # Linear damper
    f4 = x ** 2 * v          # Nonlinear damping (Van der Pol style)
    
    # Time-dependent terms (Forcing and harmonics)
    f5 = np.sin(t)           # 1st harmonic sine
    f6 = np.cos(t)           # 1st harmonic cosine
    f7 = np.sin(2 * t)       # 2nd harmonic sine
    f8 = np.cos(2 * t)       # 2nd harmonic cosine
    f9 = np.sin(3 * t)       # 3rd harmonic sine
    
    # Stack features
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # Linear combination
    out = F @ params
    
    if out.ndim != 1:
        out = out.reshape(-1)
        
    return out
