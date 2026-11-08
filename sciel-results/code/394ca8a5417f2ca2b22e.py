"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-Informed Nonlinear Oscillator Model with coupled time-state terms.
    
    Basis (10 terms) chosen to capture driven nonlinear oscillator dynamics:
    1. 1 (bias)
    2. sin(t) - periodic drive
    3. cos(t) - periodic drive (phase)
    4. x - linear restoring
    5. v - linear damping
    6. x^2 - quadratic stiffness
    7. x*v - velocity-dependent damping (Rayleigh)
    8. sin(t)*x - parametric excitation (time-modulated stiffness)
    9. cos(t)*v - parametric damping (time-modulated damping)
    10. x^3 - Duffing cubic nonlinearity
    
    This captures: periodic forcing, linear + nonlinear restoring forces,
    linear + nonlinear damping, and parametric (time-dependent) couplings.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    v = np.asarray(v)
    
    st = np.sin(t)
    ct = np.cos(t)
    
    features = np.column_stack((
        np.ones_like(t),  # 1: bias
        st,                # 2: sin(t)
        ct,                # 3: cos(t)
        x,                 # 4: x
        v,                 # 5: v
        x * x,             # 6: x^2
        x * v,             # 7: x*v (damping)
        st * x,            # 8: sin(t)*x (parametric)
        ct * v,            # 9: cos(t)*v (parametric)
        x * x * x          # 10: x^3 (Duffing)
    ))
    
    return features @ params

# EVOLVE-BLOCK-END
